from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

import extreme_locked_probability_v34 as v34

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

BULL_START = pd.Timestamp("2024-09-01").date()
MIN_BAR = 3
MAX_BAR = 35
LOCK_MIN_PCT = 0.0025
LOCK_ATR_MULT = 0.25
NEAR_EXTREME_MAX = 0.008

TEST_PERIODS = [
    ("2025_JAN_FEB", pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-02-28").date()),
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

FEATURES = list(v34.FEATURES)
BOOL_FEATURES = set(v34.BOOL_FEATURES)


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_frame():
    # Reuse all no-future intraday / regime features, but replace the old
    # close-referenced lock labels with running-extreme-referenced labels.
    z = v34.build_frame().copy().sort_values(["day", "bar_idx"]).reset_index(drop=True)
    atr = _num(z.atr_pct).fillna(0.008)
    tol = np.maximum(LOCK_MIN_PCT, LOCK_ATR_MULT * atr.to_numpy(dtype=float))
    z["running_lock_tolerance"] = tol

    # Correct objective:
    # TOP_LOCKED: the running high already observed by decision time will not
    # later be materially exceeded. BOTTOM_LOCKED is symmetric to running low.
    z["top_locked_running"] = ((z.future_high / z.cum_high - 1.0) <= z.running_lock_tolerance).astype(float)
    z["bottom_locked_running"] = ((z.cum_low / z.future_low - 1.0) <= z.running_lock_tolerance).astype(float)
    z.loc[z.future_high.isna(), "top_locked_running"] = np.nan
    z.loc[z.future_low.isna(), "bottom_locked_running"] = np.nan

    # Tradeability is deliberately separate from lock state.
    z["gap_high"] = ((z.cum_high - z.close) / z.close).clip(lower=0)
    z["gap_low"] = ((z.close - z.cum_low) / z.close).clip(lower=0)
    z["remaining_upside_from_close"] = (z.future_high / z.close - 1.0).clip(lower=0)
    z["remaining_downside_from_close"] = (z.close / z.future_low - 1.0).clip(lower=0)
    return z


def eligible(z):
    return (
        (z.bar_idx >= MIN_BAR) & (z.bar_idx <= MAX_BAR)
        & z.top_locked_running.notna() & z.bottom_locked_running.notna()
    )


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


def first_daily(z, pcol, threshold, near_col=None):
    q = z[_num(z[pcol]) >= threshold].copy()
    if near_col is not None:
        q = q[_num(q[near_col]) <= NEAR_EXTREME_MAX]
    if q.empty:
        return q
    return q.sort_values(["day", "bar_idx"]).groupby("day", as_index=False).first()


def model_metrics(z, side):
    target = "top_locked_running" if side == "TOP" else "bottom_locked_running"
    pcol = "p_top_locked_running" if side == "TOP" else "p_bottom_locked_running"
    q = z[z[target].notna() & z[pcol].notna()].copy()
    y = q[target].astype(int)
    return {
        "rows": int(len(q)),
        "base_rate": float(y.mean()) if len(q) else None,
        "auc": float(roc_auc_score(y, q[pcol])) if len(q) and y.nunique() > 1 else None,
        "brier": float(brier_score_loss(y, q[pcol])) if len(q) else None,
    }


def signal_metrics(z, side, threshold, require_near=False):
    if side == "TOP":
        target, pcol, near_col = "top_locked_running", "p_top_locked_running", "gap_high"
    else:
        target, pcol, near_col = "bottom_locked_running", "p_bottom_locked_running", "gap_low"
    q = first_daily(z, pcol, threshold, near_col if require_near else None)
    return {
        "signal_days": int(len(q)),
        "precision": float(q[target].mean()) if len(q) else None,
        "median_probability": float(q[pcol].median()) if len(q) else None,
    }


def tradeability(z, side, threshold):
    if side == "TOP":
        pcol, near_col, target = "p_top_locked_running", "gap_high", "top_locked_running"
    else:
        pcol, near_col, target = "p_bottom_locked_running", "gap_low", "bottom_locked_running"
    q = first_daily(z, pcol, threshold, near_col)
    if side == "TOP":
        favorable = q.remaining_downside_from_close
        adverse = q.remaining_upside_from_close
    else:
        favorable = q.remaining_upside_from_close
        adverse = q.remaining_downside_from_close
    ratio = favorable / adverse.replace(0, np.nan)
    return {
        "near_extreme_signal_days": int(len(q)),
        "locked_precision": float(q[target].mean()) if len(q) else None,
        "favorable_gt_adverse_rate": float((favorable > adverse).mean()) if len(q) else None,
        "median_favorable_space": float(favorable.median()) if len(q) else None,
        "median_adverse_space": float(adverse.median()) if len(q) else None,
        "median_favorable_adverse_ratio_nonzero_adverse": float(ratio.median()) if ratio.notna().any() else None,
    }


def run_walkforward(raw):
    pieces = []
    folds = {"TOP": {}, "BOTTOM": {}}
    for name, start, end in TEST_PERIODS:
        train = raw[(raw.day >= BULL_START) & (raw.day < start) & eligible(raw)].copy()
        test = raw[(raw.day >= start) & (raw.day <= end) & eligible(raw)].copy()
        if train.day.nunique() < 60 or test.day.nunique() < 20:
            raise RuntimeError(f"insufficient fold {name}: train={train.day.nunique()}, test={test.day.nunique()}")
        test["p_top_locked_running"] = predict(fit_model(train, "top_locked_running"), test)
        test["p_bottom_locked_running"] = predict(fit_model(train, "bottom_locked_running"), test)
        test["wf_period"] = name
        pieces.append(test)
        for side in ["TOP", "BOTTOM"]:
            folds[side][name] = {
                "train_days": int(train.day.nunique()),
                "test_days": int(test.day.nunique()),
                "model": model_metrics(test, side),
                "thresholds": {
                    str(t): {
                        "all": signal_metrics(test, side, t, False),
                        "near_extreme": signal_metrics(test, side, t, True),
                        "tradeability": tradeability(test, side, t),
                    }
                    for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
                },
            }
    return pd.concat(pieces, ignore_index=True), folds


def pooled_thresholds(z, side):
    return {
        str(t): {
            "all": signal_metrics(z, side, t, False),
            "near_extreme": signal_metrics(z, side, t, True),
            "tradeability": tradeability(z, side, t),
        }
        for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    }


def main():
    raw = build_frame()
    wf, folds = run_walkforward(raw)
    report = {
        "version": "extreme-running-lock-walkforward-v37",
        "objective": "rolling probability that the running intraday high/low already observed will not later be materially broken; current price distance is scored only in T tradeability layer",
        "bull_regime_start": str(BULL_START),
        "initial_training": "2024-09..2024-12",
        "chronological_oos": "2025-01..2026-08 ten expanding walk-forward double-month folds",
        "decision_bars": "bar_idx 3..35 (09:50 through 14:00 completed 5m bars)",
        "lock_label": "TOP: future_high does not exceed current running high by more than max(0.25%,0.25x5mATR); BOTTOM symmetric to running low",
        "future_leakage": False,
        "pooled_oos": {
            "TOP": {"model": model_metrics(wf, "TOP"), "thresholds": pooled_thresholds(wf, "TOP")},
            "BOTTOM": {"model": model_metrics(wf, "BOTTOM"), "thresholds": pooled_thresholds(wf, "BOTTOM")},
        },
        "folds": folds,
        "notes": [
            "Threshold table is diagnostic only in V3.7; no threshold is promoted from the same OOS results.",
            "For real T action, require both lock confidence and current price still near the running extreme; lock accuracy and T edge are reported separately.",
        ],
    }
    wf.to_csv(RESULTS / "extreme_running_lock_walkforward_v37_oos.csv", index=False)
    (RESULTS / "extreme_running_lock_walkforward_v37.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("EXTREME RUNNING LOCK WALKFORWARD V3.7")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
