from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DEV_END = pd.Timestamp("2026-06-30").date()
VAL_START = pd.Timestamp("2026-07-01").date()
MAX_ACTION_BAR = 43  # keep at least ~20 minutes before close for T execution
FIXED_TOL = 0.0025
ATR_TOL_MULT = 0.25
MIN_GROUP = 20

RAW_BINS = [-np.inf, 1.50, 2.00, 2.50, 3.00, 3.50, np.inf]
RAW_LABELS = ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", ">=3.5"]


def time_bucket(bar_idx: int) -> str:
    if bar_idx <= 8:
        return "09:50-10:15"
    if bar_idx <= 17:
        return "10:20-11:00"
    if bar_idx <= 23:
        return "11:05-11:30"
    if bar_idx <= 31:
        return "13:05-13:40"
    if bar_idx <= 39:
        return "13:45-14:20"
    return "14:25-14:40"


def add_rolling_labels(x: pd.DataFrame) -> pd.DataFrame:
    """Create scoring-only labels at every completed 5m bar.

    A top is 'locked' at t when no later bar exceeds the signal-time close by
    more than a tolerance known at t. Bottom is symmetric. Future bars are used
    only for labels/evaluation and never enter signal features.
    """
    z = x.copy().sort_values("ts").reset_index(drop=True)
    z["day"] = z.ts.dt.date
    z["bar_idx"] = z.groupby("day").cumcount()

    # Future extrema strictly AFTER the current completed bar.
    z["future_high_after"] = z.groupby("day").high.transform(
        lambda s: s.iloc[::-1].cummax().iloc[::-1].shift(-1)
    )
    z["future_low_after"] = z.groupby("day").low.transform(
        lambda s: s.iloc[::-1].cummin().iloc[::-1].shift(-1)
    )
    z["future_high_after"] = z["future_high_after"].fillna(z.close)
    z["future_low_after"] = z["future_low_after"].fillna(z.close)

    z["remaining_upside"] = (z.future_high_after / z.close - 1.0).clip(lower=0)
    z["remaining_downside"] = (z.close / z.future_low_after - 1.0).clip(lower=0)

    atr_tol = ATR_TOL_MULT * pd.to_numeric(z.atr_pct, errors="coerce")
    z["lock_tolerance"] = np.maximum(FIXED_TOL, atr_tol.fillna(FIXED_TOL))

    z["top_locked"] = z.remaining_upside <= z.lock_tolerance
    z["bottom_locked"] = z.remaining_downside <= z.lock_tolerance

    z["gap_to_running_high"] = ((z.cum_high - z.close) / z.close).clip(lower=0)
    z["gap_to_running_low"] = ((z.close - z.cum_low) / z.close).clip(lower=0)
    z["near_top"] = z.gap_to_running_high <= z.lock_tolerance
    z["near_bottom"] = z.gap_to_running_low <= z.lock_tolerance

    # 'Tradeable' means the extreme is locked AND current close is still near
    # the relevant running extreme, avoiding late signals after the move is gone.
    z["top_tradeable"] = z.top_locked & z.near_top
    z["bottom_tradeable"] = z.bottom_locked & z.near_bottom

    z["top_structure_break"] = z.prev_low.notna() & (z.close < z.prev_low)
    z["bottom_structure_break"] = z.prev_high.notna() & (z.close > z.prev_high)

    z["top_raw_score"] = (
        z.high_score
        + 0.50 * z.top_structure_break.astype(float)
        + 0.30 * z.near_top.astype(float)
        + 0.20 * (z.ret3_accel < 0).fillna(False).astype(float)
        + 0.20 * (z.pcb_rel_chg3 < 0).fillna(False).astype(float)
        + 0.10 * (z.pcb_breadth_chg3 < 0).fillna(False).astype(float)
    )
    z["bottom_raw_score"] = (
        z.low_score
        + 0.50 * z.bottom_structure_break.astype(float)
        + 0.30 * z.near_bottom.astype(float)
        + 0.20 * (z.ret3_accel > 0).fillna(False).astype(float)
        + 0.20 * (z.pcb_rel_chg3 > 0).fillna(False).astype(float)
        + 0.10 * (z.pcb_breadth_chg3 > 0).fillna(False).astype(float)
    )

    z["time_bucket"] = z.bar_idx.map(time_bucket)
    z["eligible_realtime"] = (z.bar_idx >= 3) & (z.bar_idx <= MAX_ACTION_BAR)
    return z


def add_score_bin(z: pd.DataFrame, side: str) -> pd.DataFrame:
    out = z.copy()
    col = "top_raw_score" if side == "TOP" else "bottom_raw_score"
    out["score_bin"] = pd.cut(out[col], bins=RAW_BINS, labels=RAW_LABELS, right=False).astype(str)
    return out


def _prob_table(dev: pd.DataFrame, keys, target: str) -> dict:
    g = dev.groupby(keys, observed=True)[target].agg(["mean", "count"]).reset_index()
    return {
        tuple(row[k] for k in keys): (float(row["mean"]), int(row["count"]))
        for _, row in g.iterrows()
    }


def calibrate(dev: pd.DataFrame, val: pd.DataFrame, side: str):
    """Empirical probability calibration with hierarchical fallback.

    No validation labels are used to fit probabilities.
    """
    target = "top_locked" if side == "TOP" else "bottom_locked"
    near = "near_top" if side == "TOP" else "near_bottom"

    d = add_score_bin(dev, side)
    v = add_score_bin(val, side)

    tables = [
        (["time_bucket", "score_bin", near], _prob_table(d, ["time_bucket", "score_bin", near], target), MIN_GROUP),
        (["time_bucket", "score_bin"], _prob_table(d, ["time_bucket", "score_bin"], target), MIN_GROUP),
        (["score_bin"], _prob_table(d, ["score_bin"], target), 30),
        (["time_bucket"], _prob_table(d, ["time_bucket"], target), 40),
    ]
    global_p = float(d[target].mean()) if len(d) else 0.5

    probs = []
    sources = []
    counts = []
    for _, r in v.iterrows():
        chosen = None
        for keys, table, min_n in tables:
            key = tuple(r[k] for k in keys)
            if key in table and table[key][1] >= min_n:
                chosen = (table[key][0], "+".join(keys), table[key][1])
                break
        if chosen is None:
            chosen = (global_p, "global", len(d))
        probs.append(chosen[0])
        sources.append(chosen[1])
        counts.append(chosen[2])

    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    v[pcol] = probs
    v[pcol + "_source"] = sources
    v[pcol + "_calibration_n"] = counts
    return v


def auc_binary(y, p):
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probability_metrics(z: pd.DataFrame, side: str) -> dict:
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    near = "near_top" if side == "TOP" else "near_bottom"
    adverse = "remaining_downside" if side == "TOP" else "remaining_upside"
    residual = "remaining_upside" if side == "TOP" else "remaining_downside"

    out = {
        "rows": int(len(z)),
        "base_lock_rate": float(z[target].mean()) if len(z) else None,
        "brier": float(((z[pcol] - z[target].astype(float)) ** 2).mean()) if len(z) else None,
        "auc": auc_binary(z[target], z[pcol]) if len(z) else None,
        "thresholds": {},
    }
    for th in [0.60, 0.70, 0.80]:
        q = z[(z[pcol] >= th) & z[near]].copy()
        out["thresholds"][str(th)] = {
            "signals": int(len(q)),
            "signal_days": int(q.day.nunique()) if len(q) else 0,
            "signals_per_signal_day": float(len(q) / q.day.nunique()) if len(q) and q.day.nunique() else None,
            "lock_precision": float(q[target].mean()) if len(q) else None,
            "directional_edge_rate": float((q[adverse] > q[residual]).mean()) if len(q) else None,
            "median_reversal_room": float(q[adverse].median()) if len(q) else None,
            "median_residual_wrong_way_room": float(q[residual].median()) if len(q) else None,
            "median_reversal_to_residual_ratio": float(
                (q[adverse] / q[residual].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).median()
            ) if len(q) else None,
        }
    return out


def calibration_table(z: pd.DataFrame, side: str):
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    b = pd.cut(z[pcol], bins=[0, .5, .6, .7, .8, .9, 1.000001], include_lowest=True)
    g = z.assign(prob_band=b).groupby("prob_band", observed=True).agg(
        rows=(target, "size"), predicted=(pcol, "mean"), actual=(target, "mean")
    ).reset_index()
    g["side"] = side
    return g


def main():
    x = e14.build_scored_frame()
    z = add_rolling_labels(x)
    z = z[z.eligible_realtime].copy()

    dev = z[z.day <= DEV_END].copy()
    val0 = z[z.day >= VAL_START].copy()

    top_val = calibrate(dev, val0, "TOP")
    bot_val = calibrate(dev, val0, "BOTTOM")

    # Merge calibrated probabilities by timestamp; both are generated from the
    # same validation rows but independently calibrated.
    val = top_val.copy()
    for c in ["p_bottom_locked", "p_bottom_locked_source", "p_bottom_locked_calibration_n"]:
        val[c] = bot_val[c].values

    report = {
        "version": "rolling-extreme-probability-v22",
        "status": "development_calibrated_validation_diagnostic_not_frozen",
        "primary_objective": (
            "At every completed 5-minute bar, estimate P(top already locked) and P(bottom already locked); "
            "10:00 is not a privileged target time."
        ),
        "execution_objective": (
            "Reverse-T requires high top-lock probability while current price remains near the running high, "
            "plus favorable expected downside-vs-residual-upside. Positive-T is symmetric."
        ),
        "label_definition": {
            "top_locked": "no later high exceeds signal-time close by more than max(0.25%, 0.25x signal-time ATR%)",
            "bottom_locked": "no later low undercuts signal-time close by more than max(0.25%, 0.25x signal-time ATR%)",
            "tradeable_guard": "signal-time close must still be within the same tolerance of the running extreme",
            "future_data": "used for labels/evaluation only",
        },
        "data": {
            "trading_days": int(z.day.nunique()),
            "development": "2026-01-01..2026-06-30",
            "validation": "2026-07-01..2026-08-31",
            "eligible_bars": "bar 3 through bar 43; approximately 09:50..14:40",
            "price_context_only_baseline": True,
            "note": "External realtime-news and full overseas priors are not yet promoted into this probability calibration.",
        },
        "validation": {
            "TOP": probability_metrics(val, "TOP"),
            "BOTTOM": probability_metrics(val, "BOTTOM"),
        },
    }

    cal = pd.concat([calibration_table(val, "TOP"), calibration_table(val, "BOTTOM")], ignore_index=True)
    val.to_csv(RESULTS / "rolling_extreme_probability_v22_validation.csv", index=False)
    cal.to_csv(RESULTS / "rolling_extreme_probability_v22_calibration.csv", index=False)
    (RESULTS / "rolling_extreme_probability_v22.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ROLLING EXTREME PROBABILITY V2.2")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("CALIBRATION")
    print(cal.to_string(index=False))


if __name__ == "__main__":
    main()
