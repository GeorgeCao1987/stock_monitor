from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

import event_engine_v14 as e14
import t_edge_daily_regime_model_v29 as v29
import extreme_pattern_mining_v30 as v30

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DEV_START = pd.Timestamp("2025-03-01").date()
DEV_END = pd.Timestamp("2026-08-31").date()
HOLDOUT_START = pd.Timestamp("2025-01-01").date()
HOLDOUT_END = pd.Timestamp("2025-02-28").date()

PERIODS = [
    ("2025_MAR_APR", pd.Timestamp("2025-03-01").date(), pd.Timestamp("2025-04-30").date()),
    ("2025_MAY_JUN", pd.Timestamp("2025-05-01").date(), pd.Timestamp("2025-06-30").date()),
    ("2025_JUL_AUG", pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-08-31").date()),
    ("2025_SEP_OCT", pd.Timestamp("2025-09-01").date(), pd.Timestamp("2025-10-31").date()),
    ("2025_NOV_DEC", pd.Timestamp("2025-11-01").date(), pd.Timestamp("2025-12-31").date()),
    ("2026_JAN_FEB", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-28").date()),
    ("2026_MAR_APR", pd.Timestamp("2026-03-01").date(), pd.Timestamp("2026-04-30").date()),
    ("2026_MAY_JUN", pd.Timestamp("2026-05-01").date(), pd.Timestamp("2026-06-30").date()),
    ("2026_JUL_AUG", pd.Timestamp("2026-07-01").date(), pd.Timestamp("2026-08-31").date()),
]

# Locked labels are the original rolling-extreme objective: from the current
# tradable close, no later material new extreme beyond a signal-time ATR tolerance.
LOCK_MIN_PCT = 0.0025
LOCK_ATR_MULT = 0.25
MIN_BAR = 3
MAX_BAR = 35
TOP_HIGH_CONF = 0.60
BOTTOM_HIGH_CONF = 0.60
NEAR_EXTREME_MAX = 0.008

FEATURES = list(v30.FEATURES) + ["gap_high", "gap_high_atr", "gap_low", "gap_low_atr"]
BOOL_FEATURES = set(v30.BOOL_FEATURES)


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_frame():
    x = e14.build_scored_frame()
    x = v30.add_past_features(x)
    d = v29.daily_context()
    x = x.merge(d, on="day", how="left")
    z, _ = v30.add_extreme_labels(x)
    z = z.sort_values(["day", "bar_idx"]).reset_index(drop=True)

    z["gap_high"] = ((z.cum_high - z.close) / z.close).clip(lower=0)
    z["gap_low"] = ((z.close - z.cum_low) / z.close).clip(lower=0)
    atr = _num(z.atr_pct).replace(0, np.nan)
    z["gap_high_atr"] = z.gap_high / atr
    z["gap_low_atr"] = z.gap_low / atr

    z["future_high"] = np.nan
    z["future_low"] = np.nan
    for day, g0 in z.groupby("day", sort=False):
        idx = g0.sort_values("bar_idx").index.to_numpy()
        highs = z.loc[idx, "high"].to_numpy(dtype=float)
        lows = z.loc[idx, "low"].to_numpy(dtype=float)
        fh = np.full(len(idx), np.nan)
        fl = np.full(len(idx), np.nan)
        for i in range(len(idx) - 1):
            fh[i] = np.max(highs[i + 1:])
            fl[i] = np.min(lows[i + 1:])
        z.loc[idx, "future_high"] = fh
        z.loc[idx, "future_low"] = fl

    tol = np.maximum(
        LOCK_MIN_PCT,
        LOCK_ATR_MULT * _num(z.atr_pct).fillna(0.008).to_numpy(dtype=float),
    )
    z["lock_tolerance"] = tol
    z["top_locked"] = ((z.future_high / z.close - 1.0) <= z.lock_tolerance).astype(float)
    z["bottom_locked"] = ((z.close / z.future_low - 1.0) <= z.lock_tolerance).astype(float)
    z.loc[z.future_high.isna(), "top_locked"] = np.nan
    z.loc[z.future_low.isna(), "bottom_locked"] = np.nan
    z["remaining_upside"] = (z.future_high / z.close - 1.0).clip(lower=0)
    z["remaining_downside"] = (z.close / z.future_low - 1.0).clip(lower=0)
    return z


def eligible(z):
    return (z.bar_idx >= MIN_BAR) & (z.bar_idx <= MAX_BAR) & z.top_locked.notna() & z.bottom_locked.notna()


def matrix(z):
    out = pd.DataFrame(index=z.index)
    for c in FEATURES:
        if c in BOOL_FEATURES:
            out[c] = z[c].fillna(False).astype(float)
        else:
            out[c] = _num(z[c]) if c in z.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def fit_model(train, target):
    X = matrix(train)
    med = X.median().fillna(0.0)
    y = train[target].astype(int)
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=20260903,
    )
    model.fit(X.fillna(med), y)
    return model, med


def predict(model_med, test):
    model, med = model_med
    return model.predict_proba(matrix(test).fillna(med))[:, 1]


def add_oof(z):
    out = z.copy()
    out["p_top_locked"] = np.nan
    out["p_bottom_locked"] = np.nan
    folds = {"TOP": {}, "BOTTOM": {}}
    dev = out[(out.day >= DEV_START) & (out.day <= DEV_END) & eligible(out)].copy()
    for name, start, end in PERIODS:
        test = dev[(dev.day >= start) & (dev.day <= end)].copy()
        train = dev[~((dev.day >= start) & (dev.day <= end))].copy()
        if test.empty or train.empty:
            raise RuntimeError(f"missing fold {name}")
        for side, target, pcol in [
            ("TOP", "top_locked", "p_top_locked"),
            ("BOTTOM", "bottom_locked", "p_bottom_locked"),
        ]:
            p = predict(fit_model(train, target), test)
            out.loc[test.index, pcol] = p
            y = test[target].astype(int)
            folds[side][name] = {
                "rows": int(len(test)),
                "base_rate": float(y.mean()),
                "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
                "brier": float(brier_score_loss(y, p)),
            }
    return out, folds


def episode_starts(z, cond):
    q = z.copy().sort_values(["day", "bar_idx"])
    q["_cond"] = pd.Series(cond, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False).astype(bool)
    return q[q._cond & (~prev)].drop(columns="_cond")


def signal_metrics(z, side, threshold):
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    target = "top_locked" if side == "TOP" else "bottom_locked"
    q = episode_starts(z, _num(z[pcol]) >= threshold)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "precision": float(q[target].mean()) if len(q) else None,
        "median_probability": float(q[pcol].median()) if len(q) else None,
    }


def period_signal_metrics(z, side, threshold):
    out = {}
    for name, start, end in PERIODS:
        q = z[(z.day >= start) & (z.day <= end)].copy()
        out[name] = signal_metrics(q, side, threshold)
    return out


def pooled_model_metrics(z, side):
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    q = z[z[pcol].notna() & z[target].notna()].copy()
    y = q[target].astype(int)
    return {
        "rows": int(len(q)),
        "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, q[pcol])) if y.nunique() > 1 else None,
        "brier": float(brier_score_loss(y, q[pcol])),
    }


def top_tradeability(z, threshold=TOP_HIGH_CONF):
    q = episode_starts(z, _num(z.p_top_locked) >= threshold)
    near = q[q.gap_high <= NEAR_EXTREME_MAX].copy()
    ratio = near.remaining_downside / near.remaining_upside.replace(0, np.nan)
    return {
        "locked_signals": int(len(q)),
        "near_high_signals": int(len(near)),
        "near_high_locked_precision": float(near.top_locked.mean()) if len(near) else None,
        "downside_gt_upside_rate": float((near.remaining_downside > near.remaining_upside).mean()) if len(near) else None,
        "median_remaining_downside": float(near.remaining_downside.median()) if len(near) else None,
        "median_remaining_upside": float(near.remaining_upside.median()) if len(near) else None,
        "median_downside_upside_ratio_nonzero_up": float(ratio.median()) if ratio.notna().any() else None,
    }


def top_gate(dev, folds, sig, periods):
    valid = [m for m in periods.values() if m["signals"] >= 3 and m["precision"] is not None]
    min_precision = min((m["precision"] for m in valid), default=None)
    fold_aucs = [m["auc"] for m in folds["TOP"].values() if m["auc"] is not None]
    return {
        "passed": bool(
            pooled_model_metrics(dev, "TOP")["auc"] >= 0.62
            and min(fold_aucs) >= 0.55
            and sig["signals"] >= 20
            and sig["precision"] is not None and sig["precision"] >= 0.80
            and len(valid) >= 4
            and min_precision is not None and min_precision >= 0.60
        ),
        "policy": "pooled AUC>=0.62; every fold AUC>=0.55; P_TOP_LOCKED>=0.60 gives >=20 independent signals and >=80% precision; >=4 periods with >=3 signals; worst such period >=60%",
        "periods_with_3plus": len(valid),
        "min_precision_3plus": min_precision,
    }


def bottom_gate(dev, folds, sig, periods):
    valid = [m for m in periods.values() if m["signals"] >= 3 and m["precision"] is not None]
    min_precision = min((m["precision"] for m in valid), default=None)
    fold_aucs = [m["auc"] for m in folds["BOTTOM"].values() if m["auc"] is not None]
    return {
        "passed": bool(
            pooled_model_metrics(dev, "BOTTOM")["auc"] >= 0.62
            and min(fold_aucs) >= 0.55
            and sig["signals"] >= 20
            and sig["precision"] is not None and sig["precision"] >= 0.80
            and len(valid) >= 4
            and min_precision is not None and min_precision >= 0.60
        ),
        "same_policy_as_top": True,
        "periods_with_3plus": len(valid),
        "min_precision_3plus": min_precision,
    }


def fit_full_dev_and_holdout(raw, side):
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    train = raw[(raw.day >= DEV_START) & (raw.day <= DEV_END) & eligible(raw)].copy()
    h = raw[(raw.day >= HOLDOUT_START) & (raw.day <= HOLDOUT_END) & eligible(raw)].copy()
    h[pcol] = predict(fit_model(train, target), h)
    return h


def main():
    raw = build_frame()
    oof, folds = add_oof(raw)
    dev = oof[(oof.day >= DEV_START) & (oof.day <= DEV_END) & eligible(oof)].copy()

    top_sig = signal_metrics(dev, "TOP", TOP_HIGH_CONF)
    bottom_sig = signal_metrics(dev, "BOTTOM", BOTTOM_HIGH_CONF)
    top_periods = period_signal_metrics(dev, "TOP", TOP_HIGH_CONF)
    bottom_periods = period_signal_metrics(dev, "BOTTOM", BOTTOM_HIGH_CONF)
    tgate = top_gate(dev, folds, top_sig, top_periods)
    bgate = bottom_gate(dev, folds, bottom_sig, bottom_periods)

    holdout = {
        "top_evaluated": False,
        "bottom_evaluated": False,
        "reason": "each side is opened only if its own development gate passes",
    }
    holdout_rows = []
    if tgate["passed"]:
        ht = fit_full_dev_and_holdout(raw, "TOP")
        if ht.day.nunique() < 20:
            raise RuntimeError(f"top holdout incomplete: {ht.day.nunique()} days")
        holdout.update({
            "top_evaluated": True,
            "days": int(ht.day.nunique()),
            "top_model": pooled_model_metrics(ht, "TOP"),
            "top_high_conf": signal_metrics(ht, "TOP", TOP_HIGH_CONF),
            "top_tradeability": top_tradeability(ht, TOP_HIGH_CONF),
        })
        holdout_rows.append(ht)
    if bgate["passed"]:
        hb = fit_full_dev_and_holdout(raw, "BOTTOM")
        if hb.day.nunique() < 20:
            raise RuntimeError(f"bottom holdout incomplete: {hb.day.nunique()} days")
        holdout.update({
            "bottom_evaluated": True,
            "days": int(hb.day.nunique()),
            "bottom_model": pooled_model_metrics(hb, "BOTTOM"),
            "bottom_high_conf": signal_metrics(hb, "BOTTOM", BOTTOM_HIGH_CONF),
        })
        holdout_rows.append(hb)

    report = {
        "version": "extreme-locked-probability-v34",
        "objective": "rolling P_TOP_LOCKED / P_BOTTOM_LOCKED first; T tradeability scored separately from locked probability",
        "development": "2025-03..2026-08 nine already-consumed double-month periods, leave-one-period-out",
        "reserved_holdout": "2025-01..02; each side opened only after its development gate passes",
        "label": "future material new extreme absent beyond max(0.25%, 0.25x signal-time 5m ATR) from current close",
        "decision_bars": "bar_idx 3..35 (09:50 through 14:00 completed 5m bars)",
        "features": "V3 morphology/last-impulse + VWAP/volume + running location + PCB/index context + prior-day regime; no future columns",
        "future_leakage": False,
        "pooled": {
            "TOP": pooled_model_metrics(dev, "TOP"),
            "BOTTOM": pooled_model_metrics(dev, "BOTTOM"),
        },
        "folds": folds,
        "high_confidence": {
            "TOP": {"threshold": TOP_HIGH_CONF, "pooled": top_sig, "periods": top_periods, "gate": tgate},
            "BOTTOM": {"threshold": BOTTOM_HIGH_CONF, "pooled": bottom_sig, "periods": bottom_periods, "gate": bgate},
        },
        "top_tradeability": {
            "definition": "after high-confidence TOP_LOCKED, current close still within 0.8% of running high; score whether remaining downside exceeds remaining upside",
            "development": top_tradeability(dev, TOP_HIGH_CONF),
        },
        "holdout": holdout,
    }
    dev.to_csv(RESULTS / "extreme_locked_probability_v34_dev_oof.csv", index=False)
    if holdout_rows:
        pd.concat(holdout_rows, ignore_index=True).drop_duplicates(["day", "bar_idx"]).to_csv(
            RESULTS / "extreme_locked_probability_v34_holdout.csv", index=False
        )
    (RESULTS / "extreme_locked_probability_v34.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("EXTREME LOCKED PROBABILITY V3.4")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
