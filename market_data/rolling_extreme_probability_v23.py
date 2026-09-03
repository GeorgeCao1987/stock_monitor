from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

FIXED_LOCK_TOL = 0.0025
ATR_LOCK_MULT = 0.25
TOP_EXEC_FIXED = 0.0030
TOP_EXEC_ATR_MULT = 0.35
BOTTOM_EXEC_FIXED = 0.0060
BOTTOM_EXEC_ATR_MULT = 0.75
MAX_ACTION_BAR = 43
MIN_GROUP = 15

SCORE_BINS = [-np.inf, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, np.inf]
SCORE_LABELS = ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", ">=4.5"]

PERIODS = [
    ("JAN_FEB", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-28").date()),
    ("MAR_APR", pd.Timestamp("2026-03-01").date(), pd.Timestamp("2026-04-30").date()),
    ("MAY_JUN", pd.Timestamp("2026-05-01").date(), pd.Timestamp("2026-06-30").date()),
]


def time_phase(bar_idx: int) -> str:
    if bar_idx <= 11:
        return "EARLY_0950_1030"
    if bar_idx <= 23:
        return "LATE_AM_1035_1130"
    if bar_idx <= 35:
        return "EARLY_PM_1305_1400"
    return "LATE_PM_1405_1440"


def add_v23_state(x: pd.DataFrame) -> pd.DataFrame:
    """Past-only features plus scoring-only future lock labels.

    V2.3 corrects a conceptual issue in V2.2: lock state is defined relative
    to the running extreme already observed by decision time, not the current
    executable close. Execution distance is evaluated separately.
    """
    z = x.copy().sort_values("ts").reset_index(drop=True)
    z["day"] = z.ts.dt.date
    z["bar_idx"] = z.groupby("day").cumcount()

    z["future_high_after"] = z.groupby("day").high.transform(
        lambda s: s.iloc[::-1].cummax().iloc[::-1].shift(-1)
    )
    z["future_low_after"] = z.groupby("day").low.transform(
        lambda s: s.iloc[::-1].cummin().iloc[::-1].shift(-1)
    )
    z["future_high_after"] = z["future_high_after"].fillna(z.cum_high)
    z["future_low_after"] = z["future_low_after"].fillna(z.cum_low)

    atr = pd.to_numeric(z.atr_pct, errors="coerce")
    z["lock_tolerance"] = np.maximum(FIXED_LOCK_TOL, ATR_LOCK_MULT * atr.fillna(FIXED_LOCK_TOL))

    # TRUE objective: has the extreme already observed by t become durable?
    z["future_break_above_running_high"] = (z.future_high_after / z.cum_high - 1.0).clip(lower=0)
    z["future_break_below_running_low"] = (z.cum_low / z.future_low_after - 1.0).clip(lower=0)
    z["top_locked"] = z.future_break_above_running_high <= z.lock_tolerance
    z["bottom_locked"] = z.future_break_below_running_low <= z.lock_tolerance

    # Execution quality is intentionally separate from lock truth.
    z["gap_to_running_high"] = ((z.cum_high - z.close) / z.close).clip(lower=0)
    z["gap_to_running_low"] = ((z.close - z.cum_low) / z.close).clip(lower=0)
    z["top_exec_tolerance"] = np.maximum(TOP_EXEC_FIXED, TOP_EXEC_ATR_MULT * atr.fillna(TOP_EXEC_FIXED))
    z["bottom_exec_tolerance"] = np.maximum(BOTTOM_EXEC_FIXED, BOTTOM_EXEC_ATR_MULT * atr.fillna(BOTTOM_EXEC_FIXED))
    z["top_tradeable_now"] = z.gap_to_running_high <= z.top_exec_tolerance
    z["bottom_tradeable_now"] = z.gap_to_running_low <= z.bottom_exec_tolerance

    z["remaining_upside_from_close"] = (z.future_high_after / z.close - 1.0).clip(lower=0)
    z["remaining_downside_from_close"] = (z.close / z.future_low_after - 1.0).clip(lower=0)

    # Past-only transition features.
    day = z["day"]
    rng = (z.high - z.low).replace(0, np.nan)
    z["close_pos_bar"] = ((z.close - z.low) / rng).fillna(0.5)
    z["dist_vwap_chg1"] = z.groupby(day).dist_vwap.diff()
    z["dist_vwap_chg3"] = z.groupby(day).dist_vwap.diff(3)
    z["ret1_prev"] = z.groupby(day).ret1.shift(1)
    z["new_high_prev"] = z.groupby(day).new_high.shift(1).fillna(False)
    z["new_low_prev"] = z.groupby(day).new_low.shift(1).fillna(False)
    z["no_new_high_1"] = (~z.new_high) & z.new_high_prev
    z["no_new_low_1"] = (~z.new_low) & z.new_low_prev

    z["top_structure_break"] = z.prev_low.notna() & (z.close < z.prev_low)
    z["bottom_structure_break"] = z.prev_high.notna() & (z.close > z.prev_high)
    z["failed_breakout"] = z.new_high & z.prior_cum_high.notna() & (z.close < z.prior_cum_high)
    z["failed_breakdown"] = z.new_low & z.prior_cum_low.notna() & (z.close > z.prior_cum_low)

    top_exhaust = (
        (z.vol_ratio <= 0.75).fillna(False)
        | (z.amount_ratio <= 0.75).fillna(False)
        | (z.body_shrink <= 0.70).fillna(False)
    )
    bottom_exhaust = (
        (z.vol_ratio <= 0.85).fillna(False)
        | (z.amount_ratio <= 0.85).fillna(False)
        | (z.body_shrink <= 0.70).fillna(False)
    )
    top_roll = (
        (z.ret1 <= 0).fillna(False)
        | (z.ret3_accel < 0).fillna(False)
        | z.no_new_high_1.fillna(False)
    )
    bottom_turn = (
        (z.ret1 >= 0).fillna(False)
        | (z.ret3_accel > 0).fillna(False)
        | z.no_new_low_1.fillna(False)
    )
    top_context_roll = (
        (z.pcb_rel_chg3 < 0).fillna(False)
        | (z.pcb_breadth_chg3 < 0).fillna(False)
    )
    bottom_context_turn = (
        (z.pcb_rel_chg3 > 0).fillna(False)
        | (z.pcb_breadth_chg3 > 0).fillna(False)
    )

    # HIGH remains WATCH/exhaustion-led; LOW is confirmation/stabilisation-led.
    z["top_v23_score"] = (
        z.high_score
        + 0.45 * z.high_watch.astype(float)
        + 0.45 * top_exhaust.astype(float)
        + 0.40 * top_roll.astype(float)
        + 0.30 * top_context_roll.astype(float)
        + 0.35 * z.failed_breakout.astype(float)
        + 0.30 * z.top_structure_break.astype(float)
        + 0.20 * (z.close_pos_bar <= 0.45).astype(float)
        - 0.55 * ((z.ret1 > 0.006) & (z.vol_ratio > 1.20) & (z.pcb_rel_chg3 > 0)).fillna(False).astype(float)
    )
    z["bottom_v23_score"] = (
        z.low_score
        + 0.70 * z.bottom_structure_break.astype(float)
        + 0.45 * bottom_turn.astype(float)
        + 0.35 * bottom_context_turn.astype(float)
        + 0.35 * bottom_exhaust.astype(float)
        + 0.35 * z.failed_breakdown.astype(float)
        + 0.25 * (z.dist_vwap_chg1 > 0).fillna(False).astype(float)
        + 0.20 * (z.close_pos_bar >= 0.55).astype(float)
        - 0.60 * ((z.ret1 < -0.006) & (z.vol_ratio > 1.20) & (z.pcb_rel_chg3 < 0)).fillna(False).astype(float)
    )

    z["time_phase"] = z.bar_idx.map(time_phase)
    z["eligible_realtime"] = (z.bar_idx >= 3) & (z.bar_idx <= MAX_ACTION_BAR)
    return z


def add_score_bin(z: pd.DataFrame, side: str) -> pd.DataFrame:
    out = z.copy()
    col = "top_v23_score" if side == "TOP" else "bottom_v23_score"
    out["score_bin"] = pd.cut(out[col], SCORE_BINS, labels=SCORE_LABELS, right=False).astype(str)
    return out


def _table(train: pd.DataFrame, keys, target: str):
    g = train.groupby(keys, observed=True)[target].agg(["mean", "count"]).reset_index()
    return {tuple(r[k] for k in keys): (float(r["mean"]), int(r["count"])) for _, r in g.iterrows()}


def calibrate(train: pd.DataFrame, test: pd.DataFrame, side: str) -> pd.DataFrame:
    target = "top_locked" if side == "TOP" else "bottom_locked"
    guard = "top_tradeable_now" if side == "TOP" else "bottom_tradeable_now"
    tr = add_score_bin(train, side)
    te = add_score_bin(test, side)

    # Candidate-conditioned calibration is deliberate: lock probability and
    # execution quality are separate, but live T alerts are conditioned on the
    # relevant executable region.
    tables = [
        (["time_phase", "score_bin", guard], _table(tr, ["time_phase", "score_bin", guard], target), MIN_GROUP),
        (["score_bin", guard], _table(tr, ["score_bin", guard], target), MIN_GROUP),
        (["time_phase", "score_bin"], _table(tr, ["time_phase", "score_bin"], target), 20),
        (["score_bin"], _table(tr, ["score_bin"], target), 25),
    ]
    global_p = float(tr[target].mean())
    probs, srcs, ns = [], [], []
    for _, r in te.iterrows():
        chosen = None
        for keys, tab, min_n in tables:
            key = tuple(r[k] for k in keys)
            if key in tab and tab[key][1] >= min_n:
                chosen = (tab[key][0], "+".join(keys), tab[key][1])
                break
        if chosen is None:
            chosen = (global_p, "global", len(tr))
        probs.append(chosen[0]); srcs.append(chosen[1]); ns.append(chosen[2])
    pcol = "p_top_locked_v23" if side == "TOP" else "p_bottom_locked_v23"
    te[pcol] = probs
    te[pcol + "_source"] = srcs
    te[pcol + "_n"] = ns
    return te


def episode_starts(z: pd.DataFrame, cond: pd.Series) -> pd.DataFrame:
    q = z.copy().sort_values(["day", "ts"])
    q["_cond"] = pd.Series(cond, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False)
    return q[q._cond & ~prev].drop(columns="_cond")


def signal_metrics(z: pd.DataFrame, side: str, threshold: float) -> dict:
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked_v23" if side == "TOP" else "p_bottom_locked_v23"
    guard = "top_tradeable_now" if side == "TOP" else "bottom_tradeable_now"
    reversal = "remaining_downside_from_close" if side == "TOP" else "remaining_upside_from_close"
    residual = "remaining_upside_from_close" if side == "TOP" else "remaining_downside_from_close"
    q = episode_starts(z, (z[pcol] >= threshold) & z[guard])
    if q.empty:
        return {"signals": 0, "signal_days": 0}
    ratio = q[reversal] / q[residual].replace(0, np.nan)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "lock_precision": float(q[target].mean()),
        "directional_edge_rate": float((q[reversal] > q[residual]).mean()),
        "median_reversal_room": float(q[reversal].median()),
        "median_residual_wrong_way_room": float(q[residual].median()),
        "median_reversal_to_residual_ratio": float(ratio.replace([np.inf, -np.inf], np.nan).median()) if ratio.notna().any() else None,
        "time_phases": q.time_phase.value_counts().to_dict(),
    }


def auc_binary(y, p):
    y = pd.Series(y).astype(int).to_numpy(); p = pd.Series(p).astype(float).to_numpy()
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0: return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fold_report(test: pd.DataFrame) -> dict:
    out = {}
    for side in ["TOP", "BOTTOM"]:
        target = "top_locked" if side == "TOP" else "bottom_locked"
        pcol = "p_top_locked_v23" if side == "TOP" else "p_bottom_locked_v23"
        out[side] = {
            "rows": int(len(test)),
            "base_lock_rate": float(test[target].mean()),
            "auc": auc_binary(test[target], test[pcol]),
            "brier": float(((test[pcol] - test[target].astype(float)) ** 2).mean()),
            "thresholds": {str(th): signal_metrics(test, side, th) for th in [0.50, 0.60, 0.70, 0.80]},
        }
    return out


def main():
    x = e14.build_scored_frame()
    z = add_v23_state(x)
    z = z[z.eligible_realtime].copy()

    fold_rows = []
    reports = {}
    for name, start, end in PERIODS:
        test = z[(z.day >= start) & (z.day <= end)].copy()
        train = z[~((z.day >= start) & (z.day <= end))].copy()
        if test.empty or train.empty:
            continue
        top = calibrate(train, test, "TOP")
        bot = calibrate(train, test, "BOTTOM")
        merged = top.copy()
        for c in ["p_bottom_locked_v23", "p_bottom_locked_v23_source", "p_bottom_locked_v23_n"]:
            merged[c] = bot[c].values
        merged["oof_period"] = name
        fold_rows.append(merged)
        reports[name] = fold_report(merged)

    if not fold_rows:
        raise RuntimeError("V2.3 development run requires data spanning at least two configured periods")
    oof = pd.concat(fold_rows, ignore_index=True).sort_values("ts")
    reports["POOLED_OOF"] = fold_report(oof)

    report = {
        "version": "rolling-extreme-probability-v23",
        "status": "development_oof_diagnostic_not_frozen",
        "objective": "Every 5m bar: probability that the running daily high/low already observed is durable; execution distance is scored separately.",
        "key_correction_vs_v22": "Lock truth is relative to running high/low, not current close.",
        "asymmetry": {
            "TOP": "WATCH/exhaustion-led with tighter execution-distance guard",
            "BOTTOM": "confirmation/stabilisation-led with wider execution-distance guard",
        },
        "future_leakage": "Future extrema are labels only. All score features are available by decision time.",
        "development_method": "Three two-month leave-one-period-out folds across 2026 Jan-Jun; no Jul-Aug tuning.",
        "folds": reports,
    }
    oof.to_csv(RESULTS / "rolling_extreme_probability_v23_oof.csv", index=False)
    (RESULTS / "rolling_extreme_probability_v23.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ROLLING EXTREME PROBABILITY V2.3")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
