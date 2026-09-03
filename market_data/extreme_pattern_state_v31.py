from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

import extreme_pattern_mining_v30 as v30

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

FORM_FEATURES = [f for f in v30.FEATURES if f not in {"pos_in_range", "new_high", "new_low"}]
WATCH_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
EXEC_TARGET = 0.008
EXEC_STOP = 0.008
EXEC_BARS = 12
MAX_EXEC_BAR = 35


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def matrix(z: pd.DataFrame):
    x = pd.DataFrame(index=z.index)
    for c in FORM_FEATURES:
        if c in v30.BOOL_FEATURES:
            x[c] = z[c].fillna(False).astype(float)
        else:
            x[c] = _num(z[c]) if c in z.columns else 0.0
    return x.replace([np.inf, -np.inf], np.nan)


def fit_model(train: pd.DataFrame, target: str):
    X = matrix(train)
    med = X.median().fillna(0.0)
    X = X.fillna(med)
    y = train[target].astype(int)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=7,
        min_samples_leaf=35,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=20260903,
    )
    model.fit(X, y)
    return model, med


def predict(model_med, test: pd.DataFrame):
    model, med = model_med
    return model.predict_proba(matrix(test).fillna(med))[:, 1]


def prepare():
    rows, catalog = v30.build_frame()
    z = rows.copy().sort_values(["day", "bar_idx"]).reset_index(drop=True)
    z["top_forming_15m"] = ((z.bars_to_final_high >= 0) & (z.bars_to_final_high <= 3)).astype(int)
    z["bottom_forming_15m"] = ((z.bars_to_final_low >= 0) & (z.bars_to_final_low <= 3)).astype(int)
    z["top_zone"] = (z.pos_in_range >= 0.72) | z.new_high.fillna(False)
    z["bottom_zone"] = (z.pos_in_range <= 0.28) | z.new_low.fillna(False)
    z["gap_high"] = ((z.cum_high - z.close) / z.close).clip(lower=0)
    z["gap_low"] = ((z.close - z.cum_low) / z.close).clip(lower=0)
    return z, catalog


def add_oof_probabilities(z: pd.DataFrame):
    out = z.copy()
    out["p_top_forming_15m"] = 0.0
    out["p_bottom_forming_15m"] = 0.0
    fold_metrics = {"TOP": [], "BOTTOM": []}
    for name, start, end in v30.PERIODS:
        for side, zone, target, pcol in [
            ("TOP", "top_zone", "top_forming_15m", "p_top_forming_15m"),
            ("BOTTOM", "bottom_zone", "bottom_forming_15m", "p_bottom_forming_15m"),
        ]:
            train = out[(out.day < start) | (out.day > end)]
            train = train[train[zone] & (train.bar_idx >= 3) & (train.bar_idx <= 44)].copy()
            test_mask = (out.day >= start) & (out.day <= end) & out[zone] & (out.bar_idx >= 3) & (out.bar_idx <= 44)
            test = out[test_mask].copy()
            model_med = fit_model(train, target)
            p = predict(model_med, test)
            out.loc[test.index, pcol] = p
            y = test[target].astype(int)
            fold_metrics[side].append({
                "period": name,
                "candidate_rows": int(len(test)),
                "positives": int(y.sum()),
                "base_rate": float(y.mean()),
                "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
                "brier": float(brier_score_loss(y, p)),
            })
    return out, fold_metrics


def episode_starts(z: pd.DataFrame, cond: pd.Series):
    q = z.copy().sort_values(["day", "bar_idx"])
    q["_cond"] = cond.reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False)
    return q[q._cond & (~prev)].drop(columns="_cond")


def formation_metrics(z: pd.DataFrame, side: str, threshold: float):
    if side == "TOP":
        pcol, target, zone = "p_top_forming_15m", "top_forming_15m", "top_zone"
    else:
        pcol, target, zone = "p_bottom_forming_15m", "bottom_forming_15m", "bottom_zone"
    base = z[z[zone].fillna(False) & (z.bar_idx >= 3) & (z.bar_idx <= 44)].copy()
    q = episode_starts(base, base[pcol] >= threshold)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "precision": float(q[target].mean()) if len(q) else None,
        "median_probability": float(q[pcol].median()) if len(q) else None,
    }


def choose_watch_threshold(z: pd.DataFrame, side: str):
    candidates = []
    for th in WATCH_THRESHOLDS:
        per = []
        for name, start, end in v30.PERIODS:
            x = z[(z.day >= start) & (z.day <= end)]
            per.append({"period": name, **formation_metrics(x, side, th)})
        pooled = formation_metrics(z, side, th)
        valid = [x for x in per if x["signals"] >= 8 and x["precision"] is not None]
        ps = [x["precision"] for x in valid]
        candidates.append({
            "threshold": th,
            "pooled": pooled,
            "periods": per,
            "periods_with_8plus": len(valid),
            "min_precision_8plus": min(ps) if ps else None,
            "median_precision_8plus": float(np.median(ps)) if ps else None,
        })
    eligible = [c for c in candidates if (
        c["periods_with_8plus"] == 4
        and c["min_precision_8plus"] is not None and c["min_precision_8plus"] >= 0.25
        and c["median_precision_8plus"] >= 0.35
        and c["pooled"]["signals"] >= 60
        and c["pooled"]["precision"] is not None and c["pooled"]["precision"] >= 0.35
    )]
    chosen = sorted(
        eligible,
        key=lambda c: (c["min_precision_8plus"], c["median_precision_8plus"], c["pooled"]["precision"], c["pooled"]["signals"]),
        reverse=True,
    )[0] if eligible else None
    return {"candidates": candidates, "chosen": chosen}


def add_first_passage(z: pd.DataFrame):
    x = z.copy().sort_values(["day", "bar_idx"]).reset_index(drop=True)
    pos, rev = [], []
    for i, r in x.iterrows():
        if int(r.bar_idx) > MAX_EXEC_BAR:
            pos.append(np.nan); rev.append(np.nan); continue
        fut = x.iloc[i + 1:i + 1 + EXEC_BARS]
        fut = fut[fut.day == r.day]
        if len(fut) < EXEC_BARS:
            pos.append(np.nan); rev.append(np.nan); continue

        p = 0
        for _, f in fut.iterrows():
            up = float(f.high / r.close - 1.0)
            down = float(r.close / f.low - 1.0)
            if up >= EXEC_TARGET or down >= EXEC_STOP:
                p = int(up >= EXEC_TARGET and down < EXEC_STOP)
                break
        pos.append(p)

        rr = 0
        for _, f in fut.iterrows():
            down = float(r.close / f.low - 1.0)
            up = float(f.high / r.close - 1.0)
            if down >= EXEC_TARGET or up >= EXEC_STOP:
                rr = int(down >= EXEC_TARGET and up < EXEC_STOP)
                break
        rev.append(rr)
    x["positive_execution"] = pos
    x["reverse_execution"] = rev
    return x


def execution_metrics(z: pd.DataFrame, cond: pd.Series, target: str):
    q = episode_starts(z, cond)
    q = q[q[target].notna()].copy()
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "first_passage_win": float(q[target].mean()) if len(q) else None,
        "median_gap_high": float(q.gap_high.median()) if len(q) else None,
        "median_gap_low": float(q.gap_low.median()) if len(q) else None,
    }


def evaluate_state_machine(z: pd.DataFrame, top_watch: dict, bottom_watch: dict):
    x = add_first_passage(z)
    x["recent_high_no_new"] = (x.new_high_count3 >= 1) & (~x.new_high.fillna(False))
    x["recent_low_no_new"] = (x.new_low_count3 >= 1) & (~x.new_low.fillna(False))

    top_th = top_watch["threshold"] if top_watch else 0.45
    bottom_th = bottom_watch["threshold"] if bottom_watch else 0.45
    x["top_watch_recent"] = x.groupby("day").p_top_forming_15m.transform(lambda s: s.shift(0).rolling(4, min_periods=1).max()) >= top_th
    x["bottom_watch_recent"] = x.groupby("day").p_bottom_forming_15m.transform(lambda s: s.shift(0).rolling(4, min_periods=1).max()) >= bottom_th

    atr = _num(x.atr_pct).fillna(0.008)
    x["top_exec_tol"] = np.minimum(0.008, np.maximum(0.004, 0.50 * atr))
    x["bottom_exec_tol"] = np.minimum(0.012, np.maximum(0.006, 0.75 * atr))

    x["reverse_execute_v31"] = (
        (x.bar_idx >= 3) & (x.bar_idx <= MAX_EXEC_BAR)
        & x.recent_high_no_new
        & (x.gap_high <= x.top_exec_tol)
        & (x.dist_vwap_chg1 < 0)
        & x.top_watch_recent
    )

    x["positive_confirm_v31"] = (
        (x.bar_idx >= 3) & (x.bar_idx <= MAX_EXEC_BAR)
        & x.recent_low_no_new
        & (x.gap_low <= x.bottom_exec_tol)
        & (x.pos_in_range > 0.055)
        & x.bottom_watch_recent
    )
    x["positive_execute_v31"] = (
        x.positive_confirm_v31
        & (x.close_pos_bar >= 0.55)
        & (x.dist_vwap_chg1 > 0)
    )

    report = {"periods": {}, "pooled": {}}
    for name, start, end in v30.PERIODS:
        q = x[(x.day >= start) & (x.day <= end)].copy()
        report["periods"][name] = {
            "REVERSE_EXECUTE": execution_metrics(q, q.reverse_execute_v31, "reverse_execution"),
            "POSITIVE_CONFIRM": execution_metrics(q, q.positive_confirm_v31, "positive_execution"),
            "POSITIVE_EXECUTE": execution_metrics(q, q.positive_execute_v31, "positive_execution"),
        }
    report["pooled"] = {
        "REVERSE_EXECUTE": execution_metrics(x, x.reverse_execute_v31, "reverse_execution"),
        "POSITIVE_CONFIRM": execution_metrics(x, x.positive_confirm_v31, "positive_execution"),
        "POSITIVE_EXECUTE": execution_metrics(x, x.positive_execute_v31, "positive_execution"),
    }
    return x, report


def main():
    z, catalog = prepare()
    z, folds = add_oof_probabilities(z)
    top_watch = choose_watch_threshold(z, "TOP")
    bottom_watch = choose_watch_threshold(z, "BOTTOM")
    state_rows, state_report = evaluate_state_machine(z, top_watch["chosen"], bottom_watch["chosen"])

    report = {
        "version": "extreme-pattern-state-v31",
        "development_period": "2026-01..08 only; 160 complete days; September remains untouched",
        "core_discovery": "final highs/lows are frequently preceded by a last impulse; execution should wait for transition failure/recovery rather than trade the impulse itself",
        "candidate_definition": {
            "TOP": "pos_in_range>=0.72 or new_high",
            "BOTTOM": "pos_in_range<=0.28 or new_low",
        },
        "formation_model": "OOF HistGradientBoosting on morphology/context features with direct location flags removed",
        "fold_metrics": folds,
        "watch_thresholds": {"TOP": top_watch, "BOTTOM": bottom_watch},
        "state_machine": state_report,
        "execution_label": "within next 12 completed 5m bars: +/-0.8% first passage; same-bar both hit is conservative failure",
        "future_leakage": False,
        "reserved_holdout": "2026-09 onward",
    }
    state_rows.to_csv(RESULTS / "extreme_pattern_state_v31_rows.csv", index=False)
    (RESULTS / "extreme_pattern_state_v31.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTREME PATTERN STATE V3.1")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
