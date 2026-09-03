from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

import event_engine_v14 as e14
import rolling_extreme_probability_v23 as v23

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET_MOVE = 0.008
STOP_MOVE = 0.008
EXEC_HORIZON_BARS = 12
MAX_ACTION_BAR = 35  # through ~14:00; guarantees a full 60 trading minutes

PERIODS = [
    ("2025_MAY_JUN", "2025-05-01", "2025-06-30"),
    ("2025_JUL_AUG", "2025-07-01", "2025-08-31"),
    ("2025_SEP_OCT", "2025-09-01", "2025-10-31"),
    ("2025_NOV_DEC", "2025-11-01", "2025-12-31"),
    ("2026_JAN_FEB", "2026-01-01", "2026-02-28"),
    ("2026_MAR_APR", "2026-03-01", "2026-04-30"),
    ("2026_MAY_JUN", "2026-05-01", "2026-06-30"),
    ("2026_JUL_AUG", "2026-07-01", "2026-08-31"),
]

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

FEATURES = [
    "bar_idx", "time_norm", "is_late_am", "is_pm",
    "open15_target_ret", "open15_pcb_ret", "open15_index_ret", "open15_breadth",
    "running_range_pct", "running_range_atr",
    "ret_from_open", "ret1", "ret3", "ret6", "ret3_accel",
    "dist_vwap", "dist_vwap_chg1", "dist_vwap_chg3",
    "pos_in_range", "close_pos_bar", "gap_to_running_high", "gap_to_running_low",
    "dist_to_high_atr", "dist_to_low_atr", "atr_pct",
    "vol_ratio", "amount_ratio", "body_shrink", "upper_wick_ratio", "lower_wick_ratio",
    "pcb_ret", "index_ret", "pcb_rel", "pcb_rel_chg3", "pcb_up_breadth", "pcb_breadth_chg3",
    "target_rel_pcb", "target_rel_index",
    "trend_down_score", "trend_up_score", "bottom_turn_score", "top_roll_score",
    "oversold_1", "oversold_2", "oversold_3", "oversold_4",
    "overbought_1", "overbought_2", "overbought_3", "overbought_4",
    "below_vwap_05", "below_vwap_10", "below_vwap_20",
    "above_vwap_05", "above_vwap_10", "above_vwap_20",
    "low_score", "high_score", "bottom_v23_score", "top_v23_score",
    "new_low", "new_high", "no_new_low_1", "no_new_high_1",
    "failed_breakdown", "failed_breakout", "bottom_structure_break", "top_structure_break",
    "bottom_tradeable_now", "top_tradeable_now",
]

BOOL_FEATURES = {
    "new_low", "new_high", "no_new_low_1", "no_new_high_1",
    "failed_breakdown", "failed_breakout", "bottom_structure_break", "top_structure_break",
    "bottom_tradeable_now", "top_tradeable_now",
}

TARGETS = {
    "POSITIVE_OPPORTUNITY": "positive_opportunity",
    "POSITIVE_EXECUTION": "positive_execution",
    "REVERSE_OPPORTUNITY": "reverse_opportunity",
    "REVERSE_EXECUTION": "reverse_execution",
}


def _n(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_first_passage(z: pd.DataFrame) -> pd.DataFrame:
    """Create scoring-only execution labels on the FULL 48-bar day.

    This function must run before filtering action bars; otherwise later bars
    needed by the 60-minute scoring window would be truncated.
    """
    x = z.copy().sort_values(["day", "ts"]).reset_index(drop=True)
    p_out, r_out = [], []
    for i, r in x.iterrows():
        future = x.iloc[i + 1:i + 1 + EXEC_HORIZON_BARS]
        future = future[future.day == r.day]

        pos = 0
        for _, f in future.iterrows():
            up = float(f.high / r.close - 1.0)
            down = float(r.close / f.low - 1.0)
            if up >= TARGET_MOVE or down >= STOP_MOVE:
                pos = int(up >= TARGET_MOVE and down < STOP_MOVE)
                break
        p_out.append(pos)

        rev = 0
        for _, f in future.iterrows():
            down = float(r.close / f.low - 1.0)
            up = float(f.high / r.close - 1.0)
            if down >= TARGET_MOVE or up >= STOP_MOVE:
                rev = int(down >= TARGET_MOVE and up < STOP_MOVE)
                break
        r_out.append(rev)
    x["positive_execution"] = p_out
    x["reverse_execution"] = r_out
    return x


def add_regime_features(z: pd.DataFrame) -> pd.DataFrame:
    """Past-only regime features built on the full intraday chronology."""
    x = z.copy().sort_values(["day", "ts"]).reset_index(drop=True)
    day = x.day
    x["time_norm"] = x.bar_idx / 47.0
    x["is_late_am"] = x.time_phase.eq("LATE_AM_1035_1130").astype(float)
    x["is_pm"] = x.time_phase.isin(["EARLY_PM_1305_1400", "LATE_PM_1405_1440"]).astype(float)

    # bar 2 is ~09:45. Because this is computed BEFORE eligible-bar filtering,
    # every decision bar >=3 uses only an already-completed opening snapshot.
    for src, dst in [
        ("ret_from_open", "open15_target_ret"),
        ("pcb_ret", "open15_pcb_ret"),
        ("index_ret", "open15_index_ret"),
        ("pcb_up_breadth", "open15_breadth"),
    ]:
        x[dst] = x.groupby(day)[src].transform(lambda s: s.iloc[2] if len(s) > 2 else np.nan)

    x["running_range_pct"] = (x.cum_high / x.cum_low - 1.0).clip(lower=0, upper=0.20)
    x["running_range_atr"] = (x.running_range_pct / _n(x.atr_pct).clip(lower=0.001)).clip(0, 20)
    x["target_rel_pcb"] = _n(x.ret_from_open) - _n(x.pcb_ret)
    x["target_rel_index"] = _n(x.ret_from_open) - _n(x.index_ret)

    x["trend_down_score"] = (
        (_n(x.ret1) < -0.005).astype(float)
        + (_n(x.ret3) < -0.012).astype(float)
        + (_n(x.ret6) < -0.018).astype(float)
        + (_n(x.dist_vwap_chg1) < -0.003).astype(float)
        + (_n(x.pcb_rel_chg3) < -0.005).astype(float)
        + (_n(x.pcb_breadth_chg3) < 0).astype(float)
    )
    x["trend_up_score"] = (
        (_n(x.ret1) > 0.005).astype(float)
        + (_n(x.ret3) > 0.012).astype(float)
        + (_n(x.ret6) > 0.018).astype(float)
        + (_n(x.dist_vwap_chg1) > 0.003).astype(float)
        + (_n(x.pcb_rel_chg3) > 0.005).astype(float)
        + (_n(x.pcb_breadth_chg3) > 0).astype(float)
    )
    x["bottom_turn_score"] = (
        (_n(x.ret1) >= 0).astype(float)
        + (_n(x.dist_vwap_chg1) > 0).astype(float)
        + (_n(x.pcb_rel_chg3) > 0).astype(float)
        + (_n(x.pcb_breadth_chg3) > 0).astype(float)
        + x.failed_breakdown.fillna(False).astype(float)
        + x.bottom_structure_break.fillna(False).astype(float)
    )
    x["top_roll_score"] = (
        (_n(x.ret1) <= 0).astype(float)
        + (_n(x.dist_vwap_chg1) < 0).astype(float)
        + (_n(x.pcb_rel_chg3) < 0).astype(float)
        + (_n(x.pcb_breadth_chg3) < 0).astype(float)
        + x.failed_breakout.fillna(False).astype(float)
        + x.top_structure_break.fillna(False).astype(float)
    )

    for p in [0.01, 0.02, 0.03, 0.04]:
        k = int(p * 100)
        x[f"oversold_{k}"] = (-_n(x.ret_from_open) - p).clip(lower=0, upper=0.12)
        x[f"overbought_{k}"] = (_n(x.ret_from_open) - p).clip(lower=0, upper=0.12)
    for p, suffix in [(0.005, "05"), (0.010, "10"), (0.020, "20")]:
        x[f"below_vwap_{suffix}"] = (-_n(x.dist_vwap) - p).clip(lower=0, upper=0.08)
        x[f"above_vwap_{suffix}"] = (_n(x.dist_vwap) - p).clip(lower=0, upper=0.08)
    return x


def build_frame() -> pd.DataFrame:
    x = e14.build_scored_frame()
    z = v23.add_v23_state(x)

    # CRITICAL ORDER: labels and opening-regime snapshot are built from the
    # complete 48-bar day. Only after that do we restrict decision times.
    z = add_first_passage(z)
    z["positive_opportunity"] = (
        (z.remaining_upside_from_close >= TARGET_MOVE)
        & (z.remaining_upside_from_close > z.remaining_downside_from_close)
    ).astype(int)
    z["reverse_opportunity"] = (
        (z.remaining_downside_from_close >= TARGET_MOVE)
        & (z.remaining_downside_from_close > z.remaining_upside_from_close)
    ).astype(int)
    z = add_regime_features(z)
    z = z[(z.bar_idx >= 3) & (z.bar_idx <= MAX_ACTION_BAR)].copy()
    return z


def matrix(z: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=z.index)
    for c in FEATURES:
        if c in BOOL_FEATURES:
            out[c] = z[c].fillna(False).astype(float)
        else:
            out[c] = _n(z[c]) if c in z.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def fit_model(train: pd.DataFrame, target: str):
    X = matrix(train)
    med = X.median().fillna(0.0)
    X = X.fillna(med)
    y = train[target].astype(int)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=9,
        min_samples_leaf=45,
        l2_regularization=6.0,
        early_stopping=False,
        random_state=20260903,
    )
    model.fit(X, y)
    return model, med


def predict(model_med, test: pd.DataFrame) -> np.ndarray:
    model, med = model_med
    X = matrix(test).fillna(med)
    return model.predict_proba(X)[:, 1]


def episode_starts(z: pd.DataFrame, cond: pd.Series) -> pd.DataFrame:
    q = z.copy().sort_values(["day", "ts"])
    q["_cond"] = pd.Series(cond, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False)
    return q[q._cond & (~prev)].drop(columns="_cond")


def side_guard(name: str, z: pd.DataFrame) -> pd.Series:
    return z.bottom_tradeable_now.fillna(False) if name.startswith("POSITIVE") else z.top_tradeable_now.fillna(False)


def event_metrics(z: pd.DataFrame, name: str, threshold: float) -> dict:
    pcol = "p_" + name.lower()
    target = TARGETS[name]
    q = episode_starts(z, (z[pcol] >= threshold) & side_guard(name, z))
    if q.empty:
        return {"signals": 0, "signal_days": 0, "precision": None}
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "precision": float(q[target].mean()),
        "median_probability": float(q[pcol].median()),
    }


def row_metrics(z: pd.DataFrame, name: str) -> dict:
    pcol = "p_" + name.lower()
    target = TARGETS[name]
    y = z[target].astype(int)
    p = z[pcol].astype(float)
    return {
        "rows": int(len(z)),
        "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def threshold_stability(oof: pd.DataFrame, name: str) -> dict:
    candidates = []
    for th in THRESHOLDS:
        per = []
        for period, _, _ in PERIODS:
            z = oof[oof.oof_period == period]
            per.append({"period": period, **event_metrics(z, name, th)})
        pooled = event_metrics(oof, name, th)
        valid = [x for x in per if x["signals"] >= 5 and x["precision"] is not None]
        precisions = [x["precision"] for x in valid]
        candidates.append({
            "threshold": th,
            "pooled": pooled,
            "periods": per,
            "periods_with_5plus": len(valid),
            "min_precision_5plus": min(precisions) if precisions else None,
            "median_precision_5plus": float(np.median(precisions)) if precisions else None,
        })

    eligible = []
    for c in candidates:
        if (
            c["periods_with_5plus"] >= 6
            and c["min_precision_5plus"] is not None and c["min_precision_5plus"] >= 0.50
            and c["median_precision_5plus"] >= 0.60
            and c["pooled"].get("signals", 0) >= 50
            and c["pooled"].get("precision") is not None and c["pooled"]["precision"] >= 0.62
        ):
            eligible.append(c)
    chosen = None
    if eligible:
        chosen = sorted(
            eligible,
            key=lambda c: (c["median_precision_5plus"], c["min_precision_5plus"], c["pooled"]["signals"]),
            reverse=True,
        )[0]
    return {"candidates": candidates, "frozen_candidate": chosen}


def main():
    z = build_frame()
    folds = []
    fold_report = {}
    for period, start_s, end_s in PERIODS:
        start = pd.Timestamp(start_s).date(); end = pd.Timestamp(end_s).date()
        test = z[(z.day >= start) & (z.day <= end)].copy()
        train = z[~((z.day >= start) & (z.day <= end))].copy()
        if test.empty:
            raise RuntimeError(f"missing test period {period}")
        for name, target in TARGETS.items():
            test["p_" + name.lower()] = predict(fit_model(train, target), test)
        test["oof_period"] = period
        folds.append(test)
        fold_report[period] = {name: row_metrics(test, name) for name in TARGETS}

    oof = pd.concat(folds, ignore_index=True).sort_values("ts")
    pooled = {name: row_metrics(oof, name) for name in TARGETS}
    stability = {name: threshold_stability(oof, name) for name in TARGETS}

    report = {
        "version": "t-edge-regime-model-v28",
        "status": "broad_consumed_period_oof_diagnostic",
        "objective": {
            "opportunity": "later favorable room >=0.8% and greater than wrong-way room",
            "execution": "within next 60 trading minutes, +0.8% target is reached before -0.8% risk line; symmetric for reverse",
        },
        "development_only": "2025-05-01..2026-08-31, all periods already consumed by earlier model development/holdouts",
        "new_untouched_holdout_reserved": "2025-03-01..2025-04-30; do not inspect unless a V2.8 target passes freeze policy",
        "model": "HistGradientBoostingClassifier with explicit opening/day/trend/reversal regime features",
        "chronology_fix": "opening snapshot and first-passage labels are built on full 48-bar days before eligible-bar filtering",
        "future_leakage": False,
        "max_action_bar": MAX_ACTION_BAR,
        "folds": fold_report,
        "pooled": pooled,
        "threshold_stability": stability,
        "freeze_policy": ">=6/8 periods with >=5 episodes; worst such period >=50%; median >=60%; pooled >=62%; pooled signals >=50",
    }
    oof.to_csv(RESULTS / "t_edge_regime_model_v28_oof.csv", index=False)
    (RESULTS / "t_edge_regime_model_v28.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("T EDGE REGIME MODEL V2.8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
