from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

import extreme_locked_probability_v34 as v34

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

BULL_START = pd.Timestamp("2024-09-01").date()
INITIAL_TRAIN_END = pd.Timestamp("2024-12-31").date()
TOP_THRESHOLD = 0.60
BOTTOM_THRESHOLD = 0.60
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


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def first_high_conf_per_day(z, side, threshold):
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    q = z[_num(z[pcol]) >= threshold].copy()
    if q.empty:
        return q
    return q.sort_values(["day", "bar_idx"]).groupby("day", as_index=False).first()


def model_metrics(z, side):
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    q = z[z[target].notna() & z[pcol].notna()].copy()
    y = q[target].astype(int)
    return {
        "rows": int(len(q)),
        "base_rate": float(y.mean()) if len(q) else None,
        "auc": float(roc_auc_score(y, q[pcol])) if len(q) and y.nunique() > 1 else None,
        "brier": float(brier_score_loss(y, q[pcol])) if len(q) else None,
    }


def signal_metrics(z, side, threshold):
    target = "top_locked" if side == "TOP" else "bottom_locked"
    pcol = "p_top_locked" if side == "TOP" else "p_bottom_locked"
    q = first_high_conf_per_day(z, side, threshold)
    return {
        "signal_days": int(len(q)),
        "precision": float(q[target].mean()) if len(q) else None,
        "median_probability": float(q[pcol].median()) if len(q) else None,
    }


def tradeability(z, side, threshold):
    q = first_high_conf_per_day(z, side, threshold)
    if side == "TOP":
        near = q[q.gap_high <= NEAR_EXTREME_MAX].copy()
        favorable = near.remaining_downside
        adverse = near.remaining_upside
        locked = near.top_locked
    else:
        near = q[q.gap_low <= NEAR_EXTREME_MAX].copy()
        favorable = near.remaining_upside
        adverse = near.remaining_downside
        locked = near.bottom_locked
    ratio = favorable / adverse.replace(0, np.nan)
    return {
        "high_conf_days": int(len(q)),
        "near_extreme_days": int(len(near)),
        "near_extreme_locked_precision": float(locked.mean()) if len(near) else None,
        "favorable_gt_adverse_rate": float((favorable > adverse).mean()) if len(near) else None,
        "median_favorable_space": float(favorable.median()) if len(near) else None,
        "median_adverse_space": float(adverse.median()) if len(near) else None,
        "median_favorable_adverse_ratio_nonzero_adverse": float(ratio.median()) if ratio.notna().any() else None,
    }


def train_before(raw, test_start, target):
    train = raw[(raw.day >= BULL_START) & (raw.day < test_start) & v34.eligible(raw)].copy()
    if train.day.nunique() < 60:
        raise RuntimeError(f"insufficient chronological train history before {test_start}: {train.day.nunique()} days")
    return v34.fit_model(train, target), train


def run_walkforward(raw):
    pieces = []
    folds = {"TOP": {}, "BOTTOM": {}}
    for name, start, end in TEST_PERIODS:
        test = raw[(raw.day >= start) & (raw.day <= end) & v34.eligible(raw)].copy()
        if test.day.nunique() < 20:
            raise RuntimeError(f"incomplete test fold {name}: {test.day.nunique()} days")

        top_model, top_train = train_before(raw, start, "top_locked")
        bottom_model, bottom_train = train_before(raw, start, "bottom_locked")
        test["p_top_locked"] = v34.predict(top_model, test)
        test["p_bottom_locked"] = v34.predict(bottom_model, test)
        test["wf_period"] = name
        pieces.append(test)

        for side, threshold in [("TOP", TOP_THRESHOLD), ("BOTTOM", BOTTOM_THRESHOLD)]:
            folds[side][name] = {
                "train_start": str(BULL_START),
                "train_end": str(pd.Timestamp(start) - pd.Timedelta(days=1)),
                "train_days": int(top_train.day.nunique() if side == "TOP" else bottom_train.day.nunique()),
                "test_days": int(test.day.nunique()),
                "model": model_metrics(test, side),
                "high_conf_daily_latched": signal_metrics(test, side, threshold),
                "tradeability": tradeability(test, side, threshold),
            }
    return pd.concat(pieces, ignore_index=True), folds


def threshold_table(z, side):
    rows = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        s = signal_metrics(z, side, threshold)
        t = tradeability(z, side, threshold)
        rows.append({
            "threshold": threshold,
            "signal_days": s["signal_days"],
            "precision": s["precision"],
            "near_extreme_days": t["near_extreme_days"],
            "near_extreme_locked_precision": t["near_extreme_locked_precision"],
            "favorable_gt_adverse_rate": t["favorable_gt_adverse_rate"],
        })
    return rows


def summary_gate(z, folds):
    top = signal_metrics(z, "TOP", TOP_THRESHOLD)
    fold_with_signal = [
        x["high_conf_daily_latched"]["precision"]
        for x in folds["TOP"].values()
        if x["high_conf_daily_latched"]["signal_days"] >= 1
        and x["high_conf_daily_latched"]["precision"] is not None
    ]
    fold_aucs = [x["model"]["auc"] for x in folds["TOP"].values() if x["model"]["auc"] is not None]
    return {
        "passed": bool(
            model_metrics(z, "TOP")["auc"] >= 0.60
            and len(fold_aucs) >= 8
            and float(np.median(fold_aucs)) >= 0.60
            and top["signal_days"] >= 15
            and top["precision"] is not None and top["precision"] >= 0.80
            and len(fold_with_signal) >= 5
            and float(np.median(fold_with_signal)) >= 0.75
        ),
        "policy": "chronological bull-regime walk-forward: pooled OOS AUC>=0.60; median fold AUC>=0.60; >=15 daily-latched P_TOP_LOCKED>=0.60 signal days; pooled precision>=80%; >=5 test periods with signals; median signaled-period precision>=75%",
        "median_fold_auc": float(np.median(fold_aucs)) if fold_aucs else None,
        "periods_with_signals": len(fold_with_signal),
        "median_signaled_period_precision": float(np.median(fold_with_signal)) if fold_with_signal else None,
    }


def main():
    raw = v34.build_frame()
    bull = raw[(raw.day >= BULL_START) & (raw.day <= TEST_PERIODS[-1][2])].copy()
    if bull.day.nunique() < 450:
        raise RuntimeError(f"bull-regime history too short: {bull.day.nunique()} days")

    wf, folds = run_walkforward(raw)
    report = {
        "version": "extreme-locked-walkforward-v36",
        "objective": "use all data since the Sep-2024 bull-regime start without future leakage; expanding chronological training predicts each subsequent double-month block",
        "bull_regime_start": str(BULL_START),
        "initial_training": "2024-09..2024-12",
        "chronological_oos": "2025-01..2026-08, ten expanding walk-forward test periods",
        "decision_bars": "bar_idx 3..35 (09:50 through 14:00 completed 5m bars)",
        "locked_label": "no later material new extreme beyond max(0.25%, 0.25x signal-time 5m ATR) from current close",
        "event_semantics": "probability updates each 5m; performance counts only the first high-confidence lock state per trading day",
        "future_leakage": False,
        "pooled_oos": {
            "TOP": {
                "model": model_metrics(wf, "TOP"),
                "high_conf_daily_latched": signal_metrics(wf, "TOP", TOP_THRESHOLD),
                "tradeability": tradeability(wf, "TOP", TOP_THRESHOLD),
                "threshold_diagnostic": threshold_table(wf, "TOP"),
            },
            "BOTTOM": {
                "model": model_metrics(wf, "BOTTOM"),
                "high_conf_daily_latched": signal_metrics(wf, "BOTTOM", BOTTOM_THRESHOLD),
                "tradeability": tradeability(wf, "BOTTOM", BOTTOM_THRESHOLD),
                "threshold_diagnostic": threshold_table(wf, "BOTTOM"),
            },
        },
        "folds": folds,
        "top_production_candidate_gate": summary_gate(wf, folds),
        "note": "2025 Jan-Feb is no longer a special untouched holdout; it is the first chronological OOS fold. All Sep-2024 onward data belong to the current bull-regime research universe.",
    }

    wf.to_csv(RESULTS / "extreme_locked_walkforward_v36_oos.csv", index=False)
    (RESULTS / "extreme_locked_walkforward_v36.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("EXTREME LOCKED WALKFORWARD V3.6")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
