from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

import backtest_v13 as v13
import t_edge_regime_model_v28 as v28
from config import PCB_MEMBERS

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DAILY_FEATURES = [
    "target_gap", "target_prev_ret", "target_prev_range", "target_prev_close_pos", "target_mom3", "target_mom5", "target_vol5",
    "pcb_gap", "pcb_prev_ret", "pcb_prev_range", "pcb_prev_close_pos", "pcb_mom3", "pcb_mom5", "pcb_vol5", "pcb_prev_up_breadth",
    "index_gap", "index_prev_ret", "index_prev_range", "index_prev_close_pos", "index_mom3", "index_mom5", "index_vol5",
    "gap_vs_pcb", "gap_vs_index", "prev_ret_vs_pcb", "mom5_vs_pcb", "pcb_mom5_vs_index",
]
FEATURES = v28.FEATURES + DAILY_FEATURES


def daily_symbol(symbol: str) -> pd.DataFrame:
    x = v13.load_a(symbol)
    if x.empty:
        raise RuntimeError(f"missing daily source {symbol}")
    z = x.copy().sort_values("ts")
    z["day"] = z.ts.dt.date
    d = z.groupby("day", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()
    d["daily_ret"] = d.close.pct_change()
    d["gap"] = d.open / d.close.shift(1) - 1.0
    d["prev_ret"] = d.daily_ret.shift(1)
    d["range"] = d.high / d.low - 1.0
    d["close_pos"] = ((d.close - d.low) / (d.high - d.low).replace(0, np.nan)).fillna(0.5)
    d["prev_range"] = d.range.shift(1)
    d["prev_close_pos"] = d.close_pos.shift(1)
    d["mom3"] = d.close.shift(1) / d.close.shift(4) - 1.0
    d["mom5"] = d.close.shift(1) / d.close.shift(6) - 1.0
    d["vol5"] = d.daily_ret.shift(1).rolling(5, min_periods=3).std()
    return d[["day", "gap", "prev_ret", "prev_range", "prev_close_pos", "mom3", "mom5", "vol5"]]


def daily_context() -> pd.DataFrame:
    target = daily_symbol(v13.TARGET).rename(columns={
        "gap": "target_gap", "prev_ret": "target_prev_ret", "prev_range": "target_prev_range",
        "prev_close_pos": "target_prev_close_pos", "mom3": "target_mom3", "mom5": "target_mom5", "vol5": "target_vol5",
    })
    idx = daily_symbol("000001.SH").rename(columns={
        "gap": "index_gap", "prev_ret": "index_prev_ret", "prev_range": "index_prev_range",
        "prev_close_pos": "index_prev_close_pos", "mom3": "index_mom3", "mom5": "index_mom5", "vol5": "index_vol5",
    })

    members = []
    for s in PCB_MEMBERS:
        q = daily_symbol(s).copy()
        q = q.rename(columns={c: f"{c}_{s}" for c in q.columns if c != "day"})
        members.append(q.set_index("day"))
    p = pd.concat(members, axis=1).sort_index()

    def mean_field(field):
        cols = [c for c in p.columns if c.startswith(field + "_")]
        return p[cols].mean(axis=1, skipna=True)

    pcb = pd.DataFrame(index=p.index)
    pcb["pcb_gap"] = mean_field("gap")
    pcb["pcb_prev_ret"] = mean_field("prev_ret")
    pcb["pcb_prev_range"] = mean_field("prev_range")
    pcb["pcb_prev_close_pos"] = mean_field("prev_close_pos")
    pcb["pcb_mom3"] = mean_field("mom3")
    pcb["pcb_mom5"] = mean_field("mom5")
    pcb["pcb_vol5"] = mean_field("vol5")
    prev_cols = [c for c in p.columns if c.startswith("prev_ret_")]
    pcb["pcb_prev_up_breadth"] = (p[prev_cols] > 0).sum(axis=1) / p[prev_cols].notna().sum(axis=1).replace(0, np.nan)
    pcb = pcb.reset_index()

    d = target.merge(pcb, on="day", how="left").merge(idx, on="day", how="left")
    d["gap_vs_pcb"] = d.target_gap - d.pcb_gap
    d["gap_vs_index"] = d.target_gap - d.index_gap
    d["prev_ret_vs_pcb"] = d.target_prev_ret - d.pcb_prev_ret
    d["mom5_vs_pcb"] = d.target_mom5 - d.pcb_mom5
    d["pcb_mom5_vs_index"] = d.pcb_mom5 - d.index_mom5
    return d


def build_frame() -> pd.DataFrame:
    z = v28.build_frame()
    d = daily_context()
    z = z.merge(d, on="day", how="left")
    return z


def matrix(z: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=z.index)
    for c in FEATURES:
        if c in v28.BOOL_FEATURES:
            out[c] = z[c].fillna(False).astype(float)
        else:
            out[c] = pd.to_numeric(z[c], errors="coerce") if c in z.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def fit_model(train: pd.DataFrame, target: str):
    X = matrix(train)
    med = X.median().fillna(0.0)
    X = X.fillna(med)
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
    model.fit(X, y)
    return model, med


def predict(model_med, test: pd.DataFrame) -> np.ndarray:
    model, med = model_med
    return model.predict_proba(matrix(test).fillna(med))[:, 1]


def row_metrics(z: pd.DataFrame, name: str) -> dict:
    pcol = "p_" + name.lower(); target = v28.TARGETS[name]
    y = z[target].astype(int); p = z[pcol].astype(float)
    return {
        "rows": int(len(z)), "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def event_metrics(z: pd.DataFrame, name: str, threshold: float) -> dict:
    pcol = "p_" + name.lower(); target = v28.TARGETS[name]
    guard = z.bottom_tradeable_now.fillna(False) if name.startswith("POSITIVE") else z.top_tradeable_now.fillna(False)
    q = v28.episode_starts(z, (z[pcol] >= threshold) & guard)
    if q.empty:
        return {"signals": 0, "signal_days": 0, "precision": None}
    return {
        "signals": int(len(q)), "signal_days": int(q.day.nunique()),
        "precision": float(q[target].mean()), "median_probability": float(q[pcol].median()),
    }


def stability(oof: pd.DataFrame, name: str) -> dict:
    candidates = []
    for th in v28.THRESHOLDS:
        per = []
        for period, _, _ in v28.PERIODS:
            q = oof[oof.oof_period == period]
            per.append({"period": period, **event_metrics(q, name, th)})
        pooled = event_metrics(oof, name, th)
        valid = [x for x in per if x["signals"] >= 5 and x["precision"] is not None]
        ps = [x["precision"] for x in valid]
        candidates.append({
            "threshold": th, "pooled": pooled, "periods": per,
            "periods_with_5plus": len(valid),
            "min_precision_5plus": min(ps) if ps else None,
            "median_precision_5plus": float(np.median(ps)) if ps else None,
        })
    eligible = [c for c in candidates if (
        c["periods_with_5plus"] >= 6
        and c["min_precision_5plus"] is not None and c["min_precision_5plus"] >= 0.50
        and c["median_precision_5plus"] >= 0.60
        and c["pooled"].get("signals", 0) >= 50
        and c["pooled"].get("precision") is not None and c["pooled"]["precision"] >= 0.62
    )]
    chosen = sorted(eligible, key=lambda c: (c["median_precision_5plus"], c["min_precision_5plus"], c["pooled"]["signals"]), reverse=True)[0] if eligible else None
    return {"candidates": candidates, "frozen_candidate": chosen}


def main():
    z = build_frame()
    folds = []
    fold_report = {}
    for period, start_s, end_s in v28.PERIODS:
        start = pd.Timestamp(start_s).date(); end = pd.Timestamp(end_s).date()
        test = z[(z.day >= start) & (z.day <= end)].copy()
        train = z[~((z.day >= start) & (z.day <= end))].copy()
        if test.empty:
            raise RuntimeError(f"missing test period {period}")
        for name, target in v28.TARGETS.items():
            test["p_" + name.lower()] = predict(fit_model(train, target), test)
        test["oof_period"] = period
        folds.append(test)
        fold_report[period] = {name: row_metrics(test, name) for name in v28.TARGETS}

    oof = pd.concat(folds, ignore_index=True).sort_values("ts")
    report = {
        "version": "t-edge-daily-regime-model-v29",
        "status": "broad_consumed_period_oof_diagnostic",
        "development_only": "2025-05..2026-08; all periods already consumed before V2.9",
        "new_untouched_holdout_reserved": "2025-03..04, only if a target passes freeze policy",
        "added_information": "opening gaps + previous-day structure + 3/5-day momentum/volatility for target, PCB basket and Shanghai index",
        "future_leakage": False,
        "folds": fold_report,
        "pooled": {name: row_metrics(oof, name) for name in v28.TARGETS},
        "threshold_stability": {name: stability(oof, name) for name in v28.TARGETS},
        "freeze_policy": ">=6/8 periods with >=5 episodes; worst >=50%; median >=60%; pooled >=62%; >=50 pooled signals",
    }
    oof.to_csv(RESULTS / "t_edge_daily_regime_model_v29_oof.csv", index=False)
    (RESULTS / "t_edge_daily_regime_model_v29.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("T EDGE DAILY REGIME MODEL V2.9")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
