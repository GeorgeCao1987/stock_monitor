from pathlib import Path
import json
import numpy as np
import pandas as pd

import extreme_locked_probability_v34 as v34

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

TOP_THRESHOLD = 0.60
NEAR_HIGH_MAX = 0.008


def first_high_conf_day(z: pd.DataFrame):
    q = z[(pd.to_numeric(z.p_top_locked, errors="coerce") >= TOP_THRESHOLD)].copy()
    if q.empty:
        return q
    return q.sort_values(["day", "bar_idx"]).groupby("day", as_index=False).first()


def day_signal_metrics(z: pd.DataFrame):
    q = first_high_conf_day(z)
    return {
        "signal_days": int(len(q)),
        "precision_top_locked": float(q.top_locked.mean()) if len(q) else None,
        "median_probability": float(q.p_top_locked.median()) if len(q) else None,
    }


def period_day_metrics(z: pd.DataFrame):
    out = {}
    for name, start, end in v34.PERIODS:
        q = z[(z.day >= start) & (z.day <= end)].copy()
        out[name] = day_signal_metrics(q)
    return out


def tradeability(z: pd.DataFrame):
    q = first_high_conf_day(z)
    near = q[q.gap_high <= NEAR_HIGH_MAX].copy()
    ratio = near.remaining_downside / near.remaining_upside.replace(0, np.nan)
    return {
        "high_conf_days": int(len(q)),
        "near_high_days": int(len(near)),
        "near_high_locked_precision": float(near.top_locked.mean()) if len(near) else None,
        "downside_gt_upside_rate": float((near.remaining_downside > near.remaining_upside).mean()) if len(near) else None,
        "median_remaining_downside": float(near.remaining_downside.median()) if len(near) else None,
        "median_remaining_upside": float(near.remaining_upside.median()) if len(near) else None,
        "median_downside_upside_ratio_nonzero_up": float(ratio.median()) if ratio.notna().any() else None,
    }


def gate(dev: pd.DataFrame, folds: dict, pooled_model: dict, day_metrics: dict, periods: dict):
    valid = [m for m in periods.values() if m["signal_days"] >= 2 and m["precision_top_locked"] is not None]
    ps = [m["precision_top_locked"] for m in valid]
    aucs = [m["auc"] for m in folds["TOP"].values() if m["auc"] is not None]
    return {
        "passed": bool(
            pooled_model["auc"] >= 0.62
            and min(aucs) >= 0.55
            and day_metrics["signal_days"] >= 12
            and day_metrics["precision_top_locked"] is not None
            and day_metrics["precision_top_locked"] >= 0.85
            and len(valid) >= 5
            and min(ps) >= 0.50
            and float(np.median(ps)) >= 0.80
        ),
        "policy": "daily-latched event: pooled AUC>=0.62; every fold AUC>=0.55; >=12 high-confidence signal days; pooled precision>=85%; >=5 periods with >=2 signal days; worst such period>=50%; median period precision>=80%",
        "periods_with_2plus": len(valid),
        "min_precision_2plus": min(ps) if ps else None,
        "median_precision_2plus": float(np.median(ps)) if ps else None,
    }


def predict_holdout(raw: pd.DataFrame):
    train = raw[(raw.day >= v34.DEV_START) & (raw.day <= v34.DEV_END) & v34.eligible(raw)].copy()
    h = raw[(raw.day >= v34.HOLDOUT_START) & (raw.day <= v34.HOLDOUT_END) & v34.eligible(raw)].copy()
    h["p_top_locked"] = v34.predict(v34.fit_model(train, "top_locked"), h)
    return h


def main():
    raw = v34.build_frame()
    oof, folds = v34.add_oof(raw)
    dev = oof[(oof.day >= v34.DEV_START) & (oof.day <= v34.DEV_END) & v34.eligible(oof)].copy()
    pooled_model = v34.pooled_model_metrics(dev, "TOP")
    days = day_signal_metrics(dev)
    periods = period_day_metrics(dev)
    dev_gate = gate(dev, folds, pooled_model, days, periods)

    holdout = {"evaluated": False, "reason": "development gate failed; 2025 Jan-Feb preserved"}
    holdout_rows = pd.DataFrame()
    if dev_gate["passed"]:
        h = predict_holdout(raw)
        if h.day.nunique() < 20:
            raise RuntimeError(f"holdout incomplete: {h.day.nunique()} days")
        holdout = {
            "evaluated": True,
            "days": int(h.day.nunique()),
            "model": v34.pooled_model_metrics(h, "TOP"),
            "high_conf_daily_latched": day_signal_metrics(h),
            "tradeability": tradeability(h),
        }
        holdout_rows = h

    report = {
        "version": "extreme-locked-state-v35",
        "change_from_v34": "model and threshold unchanged; correct event semantics by latching the first P_TOP_LOCKED>=0.60 state per trading day instead of counting same-day re-entries as independent lock events",
        "development": "2025-03..2026-08 nine consumed double-month periods, OOF model",
        "holdout": "2025-01..02, untouched before this run and opened only if V3.5 dev gate passes",
        "future_leakage": False,
        "model": pooled_model,
        "folds": folds["TOP"],
        "development_high_conf_daily_latched": {"threshold": TOP_THRESHOLD, "pooled": days, "periods": periods},
        "development_tradeability": tradeability(dev),
        "development_gate": dev_gate,
        "bottom_side": "not promoted: V3.4 P_BOTTOM_LOCKED pooled AUC 0.576 and only 3 high-confidence episode signals",
        "holdout_result": holdout,
    }
    dev.to_csv(RESULTS / "extreme_locked_state_v35_dev_oof.csv", index=False)
    if len(holdout_rows):
        holdout_rows.to_csv(RESULTS / "extreme_locked_state_v35_holdout.csv", index=False)
    (RESULTS / "extreme_locked_state_v35.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("EXTREME LOCKED STATE V3.5")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
