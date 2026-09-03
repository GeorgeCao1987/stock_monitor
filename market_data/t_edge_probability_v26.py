from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import rolling_extreme_probability_v23 as v23

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

MIN_EDGE_MOVE = 0.008
MAX_ACTION_BAR = 43
RIDGE_L2 = 6.0
MAX_ITER = 80
PROB_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

PERIODS = [
    ("JAN_FEB", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-28").date()),
    ("MAR_APR", pd.Timestamp("2026-03-01").date(), pd.Timestamp("2026-04-30").date()),
    ("MAY_JUN", pd.Timestamp("2026-05-01").date(), pd.Timestamp("2026-06-30").date()),
]

# All features below are available at the completed decision bar. Future extrema
# are used only to create the scoring labels.
BASE_FEATURES = [
    "bar_idx",
    "ret_from_open",
    "dist_vwap",
    "dist_vwap_chg1",
    "dist_vwap_chg3",
    "ret1",
    "ret3",
    "ret6",
    "ret3_accel",
    "pos_in_range",
    "close_pos_bar",
    "vol_ratio",
    "amount_ratio",
    "body_shrink",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "pcb_rel",
    "pcb_rel_chg3",
    "pcb_up_breadth",
    "pcb_breadth_chg3",
    "gap_to_running_high",
    "gap_to_running_low",
    "high_score",
    "low_score",
    "top_v23_score",
    "bottom_v23_score",
]

BOOL_FEATURES = [
    "new_high", "new_low", "no_new_high_1", "no_new_low_1",
    "top_structure_break", "bottom_structure_break",
    "failed_breakout", "failed_breakdown",
    "high_watch", "low_watch",
    "top_tradeable_now", "bottom_tradeable_now",
]

DERIVED_FEATURES = [
    "time_norm", "morning", "late_day",
    "oversold_open", "overbought_open",
    "below_vwap", "above_vwap",
    "vwap_recovery_1", "vwap_recovery_3",
    "vwap_rollover_1", "vwap_rollover_3",
    "pcb_turn_up", "pcb_turn_down",
    "breadth_turn_up", "breadth_turn_down",
    "low_gap_atr", "high_gap_atr",
    "down_impulse", "up_impulse",
    "bottom_recovery_combo", "top_rollover_combo",
]


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_edge_state(x: pd.DataFrame) -> pd.DataFrame:
    z = v23.add_v23_state(x).copy()
    z = z[z.bar_idx.between(3, MAX_ACTION_BAR)].copy()

    # Scoring-only labels: enough room to complete a T and more favorable room
    # in the intended direction than the residual wrong-way move.
    z["positive_t_edge"] = (
        (z.remaining_upside_from_close >= MIN_EDGE_MOVE)
        & (z.remaining_upside_from_close > z.remaining_downside_from_close)
    )
    z["reverse_t_edge"] = (
        (z.remaining_downside_from_close >= MIN_EDGE_MOVE)
        & (z.remaining_downside_from_close > z.remaining_upside_from_close)
    )

    atr = _safe_num(z.atr_pct).clip(lower=0.001)
    z["time_norm"] = z.bar_idx / 47.0
    z["morning"] = (z.bar_idx <= 23).astype(float)
    z["late_day"] = (z.bar_idx >= 36).astype(float)
    z["oversold_open"] = (-_safe_num(z.ret_from_open)).clip(lower=0, upper=0.10)
    z["overbought_open"] = _safe_num(z.ret_from_open).clip(lower=0, upper=0.10)
    z["below_vwap"] = (-_safe_num(z.dist_vwap)).clip(lower=0, upper=0.06)
    z["above_vwap"] = _safe_num(z.dist_vwap).clip(lower=0, upper=0.06)
    z["vwap_recovery_1"] = _safe_num(z.dist_vwap_chg1).clip(lower=0, upper=0.03)
    z["vwap_recovery_3"] = _safe_num(z.dist_vwap_chg3).clip(lower=0, upper=0.05)
    z["vwap_rollover_1"] = (-_safe_num(z.dist_vwap_chg1)).clip(lower=0, upper=0.03)
    z["vwap_rollover_3"] = (-_safe_num(z.dist_vwap_chg3)).clip(lower=0, upper=0.05)
    z["pcb_turn_up"] = _safe_num(z.pcb_rel_chg3).clip(lower=0, upper=0.04)
    z["pcb_turn_down"] = (-_safe_num(z.pcb_rel_chg3)).clip(lower=0, upper=0.04)
    z["breadth_turn_up"] = _safe_num(z.pcb_breadth_chg3).clip(lower=0)
    z["breadth_turn_down"] = (-_safe_num(z.pcb_breadth_chg3)).clip(lower=0)
    z["low_gap_atr"] = (_safe_num(z.gap_to_running_low) / atr).clip(lower=0, upper=5)
    z["high_gap_atr"] = (_safe_num(z.gap_to_running_high) / atr).clip(lower=0, upper=5)
    z["down_impulse"] = (-_safe_num(z.ret3)).clip(lower=0, upper=0.06)
    z["up_impulse"] = _safe_num(z.ret3).clip(lower=0, upper=0.06)

    # Interaction-like features encode the observed asymmetry without future data:
    # bottom = oversold + stabilisation/recovery; top = overbought + rollover.
    z["bottom_recovery_combo"] = (
        z.oversold_open
        + z.below_vwap
        + 2.0 * z.vwap_recovery_1
        + 1.5 * z.pcb_turn_up
        + 0.01 * z.breadth_turn_up
        + 0.01 * z.bottom_structure_break.astype(float)
    )
    z["top_rollover_combo"] = (
        z.overbought_open
        + z.above_vwap
        + 2.0 * z.vwap_rollover_1
        + 1.5 * z.pcb_turn_down
        + 0.01 * z.breadth_turn_down
        + 0.01 * z.top_structure_break.astype(float)
    )
    return z


def feature_frame(z: pd.DataFrame) -> pd.DataFrame:
    cols = BASE_FEATURES + BOOL_FEATURES + DERIVED_FEATURES
    out = pd.DataFrame(index=z.index)
    for c in cols:
        if c not in z.columns:
            out[c] = 0.0
        elif c in BOOL_FEATURES:
            out[c] = z[c].fillna(False).astype(float)
        else:
            out[c] = _safe_num(z[c])
    return out


def _sigmoid(a):
    a = np.clip(a, -35, 35)
    return 1.0 / (1.0 + np.exp(-a))


def fit_logistic_ridge(train: pd.DataFrame, target: str):
    X0 = feature_frame(train)
    med = X0.median(axis=0).fillna(0.0)
    X0 = X0.fillna(med)
    mean = X0.mean(axis=0)
    std = X0.std(axis=0).replace(0, 1.0).fillna(1.0)
    Xs = ((X0 - mean) / std).clip(-8, 8).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(Xs)), Xs])
    y = train[target].astype(float).to_numpy()

    base = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = math.log(base / (1.0 - base))
    penalty = np.eye(X.shape[1]) * RIDGE_L2
    penalty[0, 0] = 0.0

    for _ in range(MAX_ITER):
        p = _sigmoid(X @ beta)
        w = np.clip(p * (1.0 - p), 1e-5, None)
        grad = X.T @ (y - p) - penalty @ beta
        h = (X.T * w) @ X + penalty
        try:
            step = np.linalg.solve(h, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(h) @ grad
        beta_new = beta + step
        if float(np.max(np.abs(step))) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    return {"beta": beta, "median": med, "mean": mean, "std": std, "features": list(X0.columns)}


def predict_model(model, test: pd.DataFrame) -> np.ndarray:
    X0 = feature_frame(test)[model["features"]]
    X0 = X0.fillna(model["median"])
    Xs = ((X0 - model["mean"]) / model["std"]).clip(-8, 8).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(Xs)), Xs])
    return _sigmoid(X @ model["beta"])


def auc_binary(y, p):
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def episode_starts(z: pd.DataFrame, cond: pd.Series) -> pd.DataFrame:
    q = z.copy().sort_values(["day", "ts"])
    q["_cond"] = pd.Series(cond, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False)
    return q[q._cond & (~prev)].drop(columns="_cond")


def edge_metrics(z: pd.DataFrame, side: str, threshold: float) -> dict:
    if side == "POSITIVE":
        pcol, target, guard = "p_positive_t_edge", "positive_t_edge", "bottom_tradeable_now"
        favorable, wrong = "remaining_upside_from_close", "remaining_downside_from_close"
    else:
        pcol, target, guard = "p_reverse_t_edge", "reverse_t_edge", "top_tradeable_now"
        favorable, wrong = "remaining_downside_from_close", "remaining_upside_from_close"

    cond = (z[pcol] >= threshold) & z[guard]
    q = episode_starts(z, cond)
    if q.empty:
        return {"signals": 0, "signal_days": 0, "success_rate": None}
    ratio = q[favorable] / q[wrong].replace(0, np.nan)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "success_rate": float(q[target].mean()),
        "move_hit_rate": float((q[favorable] >= MIN_EDGE_MOVE).mean()),
        "directional_edge_rate": float((q[favorable] > q[wrong]).mean()),
        "median_favorable_room": float(q[favorable].median()),
        "median_wrong_way_room": float(q[wrong].median()),
        "median_room_ratio": float(ratio.replace([np.inf, -np.inf], np.nan).median()) if ratio.notna().any() else None,
        "time_phases": q.time_phase.value_counts().to_dict(),
    }


def row_metrics(z: pd.DataFrame, side: str) -> dict:
    pcol = "p_positive_t_edge" if side == "POSITIVE" else "p_reverse_t_edge"
    target = "positive_t_edge" if side == "POSITIVE" else "reverse_t_edge"
    y = z[target].astype(float)
    p = z[pcol].astype(float)
    return {
        "rows": int(len(z)),
        "base_rate": float(y.mean()),
        "auc": auc_binary(y, p),
        "brier": float(((p - y) ** 2).mean()),
    }


def period_report(z: pd.DataFrame) -> dict:
    out = {}
    for side in ["POSITIVE", "REVERSE"]:
        out[side] = row_metrics(z, side)
        out[side]["thresholds"] = {str(t): edge_metrics(z, side, t) for t in PROB_THRESHOLDS}
    return out


def threshold_stability(oof: pd.DataFrame, side: str) -> dict:
    candidates = []
    for th in PROB_THRESHOLDS:
        per = []
        pooled = edge_metrics(oof, side, th)
        for name, _, _ in PERIODS:
            q = oof[oof.oof_period == name]
            m = edge_metrics(q, side, th)
            per.append({"period": name, **m})
        valid = [x for x in per if x["signals"] >= 5 and x["success_rate"] is not None]
        min_rate = min((x["success_rate"] for x in valid), default=None)
        candidates.append({
            "threshold": th,
            "pooled": pooled,
            "periods": per,
            "periods_with_5plus": len(valid),
            "min_rate_5plus": min_rate,
        })

    # Freeze only if all three periods have enough independent alerts and the
    # worst-period precision remains >=60%. Otherwise explicitly do not freeze.
    eligible = [
        c for c in candidates
        if c["periods_with_5plus"] == 3
        and c["min_rate_5plus"] is not None and c["min_rate_5plus"] >= 0.60
        and c["pooled"].get("signals", 0) >= 20
        and c["pooled"].get("success_rate") is not None and c["pooled"]["success_rate"] >= 0.65
    ]
    chosen = None
    if eligible:
        # Prefer higher worst-period precision, then more signals, then lower threshold.
        chosen = sorted(
            eligible,
            key=lambda c: (c["min_rate_5plus"], c["pooled"]["signals"], -c["threshold"]),
            reverse=True,
        )[0]
    return {"candidates": candidates, "frozen_candidate": chosen}


def main():
    x = e14.build_scored_frame()
    z = add_edge_state(x)

    folds = []
    reports = {}
    for name, start, end in PERIODS:
        test = z[(z.day >= start) & (z.day <= end)].copy()
        train = z[~((z.day >= start) & (z.day <= end))].copy()
        if test.empty or train.empty:
            continue
        pos_model = fit_logistic_ridge(train, "positive_t_edge")
        rev_model = fit_logistic_ridge(train, "reverse_t_edge")
        test["p_positive_t_edge"] = predict_model(pos_model, test)
        test["p_reverse_t_edge"] = predict_model(rev_model, test)
        test["oof_period"] = name
        folds.append(test)
        reports[name] = period_report(test)

    if len(folds) != 3:
        raise RuntimeError("V2.6 development requires all three Jan-Jun 2026 folds")

    oof = pd.concat(folds, ignore_index=True).sort_values("ts")
    reports["POOLED_OOF"] = period_report(oof)
    stability = {
        "POSITIVE": threshold_stability(oof, "POSITIVE"),
        "REVERSE": threshold_stability(oof, "REVERSE"),
    }

    report = {
        "version": "t-edge-probability-v26",
        "status": "development_oof_probability_diagnostic",
        "objective": {
            "positive": "P(future upside >=0.8% AND future upside > future downside)",
            "reverse": "P(future downside >=0.8% AND future downside > future upside)",
        },
        "method": "ridge logistic regression, separate positive/reverse models, three two-month leave-one-period-out folds",
        "important": "Extreme-lock probabilities remain separate outputs; T-edge probability is not the same as final top/bottom lock probability.",
        "features": BASE_FEATURES + BOOL_FEATURES + DERIVED_FEATURES,
        "ridge_l2": RIDGE_L2,
        "future_leakage": False,
        "development_period": "2026-01-01..2026-06-30 only",
        "forbidden_for_tuning": ["2025-07..08", "2025-09..10", "2025-11..12", "2026-07..08"],
        "freeze_policy": "All 3 folds >=5 episode starts, worst-fold success >=60%, pooled >=65%, pooled signals >=20.",
        "folds": reports,
        "threshold_stability": stability,
    }

    oof.to_csv(RESULTS / "t_edge_probability_v26_oof.csv", index=False)
    (RESULTS / "t_edge_probability_v26.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("T EDGE PROBABILITY V2.6")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
