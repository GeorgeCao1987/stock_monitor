from pathlib import Path
import json
import numpy as np
import pandas as pd

import backtest_v13 as v13
import backtest_v14 as v14
import backtest_v15 as v15
import event_engine_v14 as e14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"

FEATURES = [
    "high_score", "ret_from_open", "dist_vwap", "pos_in_range", "atr_pct",
    "vol_ratio", "amount_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "body_shrink", "ret3", "ret6", "ret3_accel", "pcb_rel",
    "pcb_rel_chg3", "pcb_breadth_chg3", "close_pos_bar", "bar_body_signed",
    "up_votes", "down_votes",
]


def auc_rank(y, x):
    z = pd.DataFrame({"y": y, "x": x}).dropna()
    if z.empty or z.y.nunique() < 2:
        return None
    n1 = int((z.y == 1).sum())
    n0 = int((z.y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = z.x.rank(method="average")
    r1 = ranks[z.y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def robust_effect(y, x):
    z = pd.DataFrame({"y": y, "x": x}).dropna()
    if z.empty or z.y.nunique() < 2:
        return None
    q = z.loc[z.y == 1, "x"]
    n = z.loc[z.y == 0, "x"]
    iqr = z.x.quantile(.75) - z.x.quantile(.25)
    if not np.isfinite(iqr) or abs(iqr) < 1e-12:
        return None
    return float((q.median() - n.median()) / iqr)


def feature_stats(w):
    out = {}
    for f in FEATURES:
        if f not in w.columns:
            continue
        z = w[["quick_confirm", f]].dropna()
        if z.empty:
            continue
        a = auc_rank(z.quick_confirm.astype(int), z[f])
        out[f] = {
            "n": int(len(z)),
            "quick_n": int(z.quick_confirm.sum()),
            "median_quick": float(z.loc[z.quick_confirm, f].median()) if z.quick_confirm.any() else None,
            "median_nonquick": float(z.loc[~z.quick_confirm, f].median()) if (~z.quick_confirm).any() else None,
            "auc_high_predicts_quick": a,
            "oriented_auc": None if a is None else float(max(a, 1-a)),
            "direction": None if a is None else ("HIGH" if a >= .5 else "LOW"),
            "robust_median_effect_iqr": robust_effect(z.quick_confirm.astype(int), z[f]),
        }
    return out


def component_stats(w):
    specs = {
        "NEW_HIGH": w.new_high,
        "UPPER_WICK_GE_035": w.upper_wick_ratio >= .35,
        "LOW_VOL_LE_090": w.vol_ratio <= .90,
        "BODY_SHRINK_LE_075": w.body_shrink <= .75,
        "RET3_DECEL_NEG": w.ret3_accel < 0,
        "PCB_REL_WEAK_SCORE": w.pcb_rel_chg3 < -.0015,
        "PCB_REL_WEAK_ANY": w.pcb_rel_chg3 < 0,
        "PCB_BREADTH_WEAK": w.pcb_breadth_chg3 < 0,
        "ABOVE_VWAP_GT_0006": w.dist_vwap > .006,
        "ATR_GT_0008": w.atr_pct > .008,
        "BAR_RED": w.bar_body_signed < 0,
        "CLOSE_IN_LOWER_HALF": w.close_pos_bar <= .50,
        "TREND_UP": w.trend_state == "UP",
        "TREND_RANGE": w.trend_state == "RANGE",
        "TREND_DOWN": w.trend_state == "DOWN",
    }
    out = {}
    for name, cond in specs.items():
        c = pd.Series(cond, index=w.index).fillna(False).astype(bool)
        p = w[c]
        a = w[~c]
        rp = float(p.quick_confirm.mean()) if len(p) else None
        ra = float(a.quick_confirm.mean()) if len(a) else None
        out[name] = {
            "present_n": int(len(p)),
            "present_quick_rate": rp,
            "absent_n": int(len(a)),
            "absent_quick_rate": ra,
            "quick_rate_diff": None if rp is None or ra is None else float(rp-ra),
        }
    return out


def build():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = v14.add_v14_features(target)
    x = v14.add_context(x)
    x = v14.score_states(x)
    x = v15.add_trend_regime(x)
    x = x.sort_values("ts").reset_index(drop=True)
    day = x.ts.dt.date
    x["bar_idx_day"] = x.groupby(day).cumcount()
    rng = (x.high - x.low).replace(0, np.nan)
    x["close_pos_bar"] = (x.close - x.low) / rng
    x["bar_body_signed"] = (x.close - x.open) / x.close
    x["prev_high"] = x.groupby(day).high.shift(1)
    x["prev_low"] = x.groupby(day).low.shift(1)

    events = e14.build_events(x)
    h = events[events.side == "HIGH"].copy().sort_values(["event_id", "ts"])
    rows = []
    idx_map = x.set_index("ts")["bar_idx_day"].to_dict()
    feature_cols = [c for c in FEATURES if c in x.columns] + [
        "new_high", "trend_state", "upper_wick_ratio", "vol_ratio", "body_shrink",
        "ret3_accel", "pcb_rel_chg3", "pcb_breadth_chg3", "dist_vwap", "atr_pct",
        "bar_body_signed", "close_pos_bar"
    ]
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_frame = x.set_index("ts")[feature_cols]

    for eid, g in h.groupby("event_id"):
        w = g[g.event_type == "WATCH_START"]
        if w.empty:
            continue
        wr = w.iloc[0]
        c = g[(g.event_type == "STRUCTURE_CONFIRM") & (g.ts >= wr.ts)]
        if c.empty:
            delay = None
        else:
            cr = c.iloc[0]
            wi = idx_map.get(wr.ts)
            ci = idx_map.get(cr.ts)
            delay = None if wi is None or ci is None else int(ci-wi)
        rec = {
            "event_id": eid,
            "watch_ts": wr.ts,
            "confirm_delay_bars": delay,
            "quick_confirm": bool(delay is not None and 1 <= delay <= 2),
            "watch_future_30m": wr.future_30m,
            "watch_mfe_30m": wr.mfe_30m,
            "watch_mae_30m": wr.mae_30m,
        }
        if wr.ts in feature_frame.index:
            fr = feature_frame.loc[wr.ts]
            for col in feature_cols:
                rec[col] = fr[col]
        rows.append(rec)
    return x, pd.DataFrame(rows)


def outcome_summary(w):
    q = w[w.quick_confirm]
    n = w[~w.quick_confirm]
    def m(z):
        if z.empty:
            return {"n": 0}
        valid = z.atr_pct.notna() & (z.atr_pct > 0)
        a = z[valid]
        return {
            "n": int(len(z)),
            "fixed_1_5": float((z.watch_future_30m >= .015).mean()),
            "atr_0_75": float((a.watch_mfe_30m >= .75*a.atr_pct).mean()) if len(a) else None,
            "directional": float((z.watch_mfe_30m > z.watch_mae_30m).mean()),
        }
    return {"quick": m(q), "nonquick": m(n)}


def main():
    x, w = build()
    report = {
        "trading_days": int(x.ts.dt.date.nunique()),
        "high_watch_events": int(len(w)),
        "quick_confirm_events": int(w.quick_confirm.sum()),
        "quick_confirm_rate": float(w.quick_confirm.mean()) if len(w) else None,
        "outcome_from_watch": outcome_summary(w),
        "univariate_features": feature_stats(w),
        "binary_components": component_stats(w),
        "method_note": "quick_confirm is a future research label only. Every predictor value is taken at WATCH_START and uses no later bar.",
    }
    w.to_csv(RESULTS / "watch_predictor_events.csv", index=False)
    (RESULTS / "watch_predictor_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("WATCH PREDICTOR DIAGNOSTICS")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
