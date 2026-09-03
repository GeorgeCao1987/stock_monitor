from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree

import event_engine_v14 as e14
import t_edge_daily_regime_model_v29 as v29

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

ANALYSIS_START = pd.Timestamp("2026-01-01").date()
ANALYSIS_END = pd.Timestamp("2026-08-31").date()
# September 2026 is deliberately not consumed by this discovery pass.

PERIODS = [
    ("JAN_FEB", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-28").date()),
    ("MAR_APR", pd.Timestamp("2026-03-01").date(), pd.Timestamp("2026-04-30").date()),
    ("MAY_JUN", pd.Timestamp("2026-05-01").date(), pd.Timestamp("2026-06-30").date()),
    ("JUL_AUG", pd.Timestamp("2026-07-01").date(), pd.Timestamp("2026-08-31").date()),
]

FEATURES = [
    "bar_idx", "ret_from_open", "ret1", "ret3", "ret6", "ret12", "ret3_accel",
    "dist_vwap", "dist_vwap_chg1", "dist_vwap_chg3",
    "pos_in_range", "close_pos_bar", "atr_pct", "vol_ratio", "amount_ratio",
    "upper_wick_ratio", "lower_wick_ratio", "body_shrink", "bar_body_abs",
    "new_high", "new_low", "new_high_count3", "new_low_count3",
    "hh_count3", "hl_count3", "lh_count3", "ll_count3",
    "pcb_ret", "index_ret", "pcb_rel", "pcb_rel_chg1", "pcb_rel_chg3",
    "pcb_up_breadth", "pcb_breadth_chg1", "pcb_breadth_chg3",
    "target_rel_pcb", "target_rel_index",
    "target_gap", "target_prev_ret", "target_prev_range", "target_prev_close_pos",
    "target_mom3", "target_mom5", "target_vol5",
    "pcb_gap", "pcb_prev_ret", "pcb_mom3", "pcb_mom5", "pcb_vol5",
    "index_gap", "index_prev_ret", "index_mom3", "index_mom5", "index_vol5",
]

BOOL_FEATURES = {"new_high", "new_low"}
TARGETS = ["pre_high_15m", "pre_low_15m", "post_high_10m", "post_low_10m"]


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def phase(bar_idx: int) -> str:
    if bar_idx <= 5:
        return "0935_1000"
    if bar_idx <= 11:
        return "1005_1030"
    if bar_idx <= 17:
        return "1035_1100"
    if bar_idx <= 23:
        return "1105_1130"
    if bar_idx <= 29:
        return "1305_1330"
    if bar_idx <= 35:
        return "1335_1400"
    if bar_idx <= 41:
        return "1405_1430"
    return "1435_1500"


def add_past_features(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy().sort_values("ts").reset_index(drop=True)
    z["day"] = z.ts.dt.date
    g = z.groupby("day", sort=False)
    z["ret12"] = g.close.pct_change(12)
    z["dist_vwap_chg1"] = g.dist_vwap.diff(1)
    z["dist_vwap_chg3"] = g.dist_vwap.diff(3)
    z["pcb_rel_chg1"] = g.pcb_rel.diff(1)
    z["pcb_breadth_chg1"] = g.pcb_up_breadth.diff(1)

    rng = (z.high - z.low).replace(0, np.nan)
    z["close_pos_bar"] = ((z.close - z.low) / rng).fillna(0.5)
    z["hh"] = z.high > g.high.shift(1)
    z["hl"] = z.low > g.low.shift(1)
    z["lh"] = z.high < g.high.shift(1)
    z["ll"] = z.low < g.low.shift(1)
    for src, dst in [
        ("new_high", "new_high_count3"), ("new_low", "new_low_count3"),
        ("hh", "hh_count3"), ("hl", "hl_count3"),
        ("lh", "lh_count3"), ("ll", "ll_count3"),
    ]:
        z[dst] = g[src].transform(lambda s: s.astype(float).rolling(3, min_periods=1).sum())

    z["target_rel_pcb"] = _num(z.ret_from_open) - _num(z.pcb_ret)
    z["target_rel_index"] = _num(z.ret_from_open) - _num(z.index_ret)
    z["phase"] = z.bar_idx.map(phase)
    return z


def add_extreme_labels(x: pd.DataFrame):
    z = x.copy().sort_values(["day", "ts"]).reset_index(drop=True)
    rows = []
    catalog = []
    for day, g0 in z.groupby("day", sort=True):
        g = g0.sort_values("ts").copy()
        if len(g) != 48:
            continue
        positions = np.arange(len(g))
        hi_pos = int(np.argmax(g.high.to_numpy(dtype=float)))
        lo_pos = int(np.argmin(g.low.to_numpy(dtype=float)))
        high_row = g.iloc[hi_pos]
        low_row = g.iloc[lo_pos]
        catalog.append({
            "day": day,
            "high_bar": hi_pos, "high_ts": high_row.ts, "high_price": float(high_row.high),
            "high_ret_from_open": float(high_row.high / g.iloc[0].open - 1.0),
            "low_bar": lo_pos, "low_ts": low_row.ts, "low_price": float(low_row.low),
            "low_ret_from_open": float(low_row.low / g.iloc[0].open - 1.0),
            "high_phase": phase(hi_pos), "low_phase": phase(lo_pos),
            "high_before_low": bool(hi_pos < lo_pos),
        })
        q = g.copy()
        q["bars_to_final_high"] = hi_pos - positions
        q["bars_to_final_low"] = lo_pos - positions
        q["bars_after_final_high"] = positions - hi_pos
        q["bars_after_final_low"] = positions - lo_pos
        q["pre_high_15m"] = ((q.bars_to_final_high >= 1) & (q.bars_to_final_high <= 3)).astype(int)
        q["pre_low_15m"] = ((q.bars_to_final_low >= 1) & (q.bars_to_final_low <= 3)).astype(int)
        q["post_high_10m"] = ((q.bars_after_final_high >= 1) & (q.bars_after_final_high <= 2)).astype(int)
        q["post_low_10m"] = ((q.bars_after_final_low >= 1) & (q.bars_after_final_low <= 2)).astype(int)
        q["near_any_extreme"] = (
            (q.bars_to_final_high.abs() <= 3) | (q.bars_to_final_low.abs() <= 3)
        )
        rows.append(q)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(catalog)


def build_frame():
    x = e14.build_scored_frame()
    x = add_past_features(x)
    d = v29.daily_context()
    x = x.merge(d, on="day", how="left")
    x = x[(x.day >= ANALYSIS_START) & (x.day <= ANALYSIS_END)].copy()
    return add_extreme_labels(x)


def matrix(z: pd.DataFrame, features=FEATURES):
    out = pd.DataFrame(index=z.index)
    for c in features:
        if c in BOOL_FEATURES:
            out[c] = z[c].fillna(False).astype(float)
        else:
            out[c] = _num(z[c]) if c in z.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def oof_tree_auc(rows: pd.DataFrame, target: str):
    folds = []
    for name, start, end in PERIODS:
        test = rows[(rows.day >= start) & (rows.day <= end)].copy()
        train = rows[~((rows.day >= start) & (rows.day <= end))].copy()
        if test.empty or train.empty:
            continue
        Xtr = matrix(train)
        med = Xtr.median().fillna(0.0)
        Xtr = Xtr.fillna(med)
        Xte = matrix(test).fillna(med)
        ytr = train[target].astype(int)
        yte = test[target].astype(int)
        model = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=60, class_weight="balanced", random_state=20260903
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, p)) if yte.nunique() > 1 else None
        folds.append({"period": name, "rows": int(len(test)), "positives": int(yte.sum()), "auc": auc})
    valid = [f["auc"] for f in folds if f["auc"] is not None]
    return {
        "folds": folds,
        "median_auc": float(np.median(valid)) if valid else None,
        "min_auc": float(np.min(valid)) if valid else None,
    }


def extract_tree_rules(rows: pd.DataFrame, target: str):
    X = matrix(rows)
    med = X.median().fillna(0.0)
    X = X.fillna(med)
    y = rows[target].astype(int)
    model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=60, class_weight="balanced", random_state=20260903
    )
    model.fit(X, y)
    names = list(X.columns)
    tree = model.tree_
    base = float(y.mean())
    rules = []

    def walk(node, conds):
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            idx = np.ones(len(X), dtype=bool)
            for feat, op, threshold in conds:
                vals = X[feat].to_numpy(dtype=float)
                idx &= vals <= threshold if op == "<=" else vals > threshold
            n = int(idx.sum())
            pos = int(y.to_numpy()[idx].sum()) if n else 0
            precision = pos / n if n else None
            rules.append({
                "conditions": [f"{f} {op} {t:.6f}" for f, op, t in conds],
                "rows": n, "positives": pos, "precision": precision,
                "lift": precision / base if n and base > 0 else None,
            })
            return
        feat = names[tree.feature[node]]
        th = float(tree.threshold[node])
        walk(tree.children_left[node], conds + [(feat, "<=", th)])
        walk(tree.children_right[node], conds + [(feat, ">", th)])

    walk(0, [])
    rules = [r for r in rules if r["rows"] >= 60 and r["precision"] is not None]
    rules.sort(key=lambda r: (r["lift"], r["positives"]), reverse=True)
    return {"base_rate": base, "top_rules": rules[:8]}


def quantile_lifts(rows: pd.DataFrame, target: str):
    y = rows[target].astype(int)
    base = float(y.mean())
    out = []
    for f in FEATURES:
        s = _num(rows[f]) if f in rows.columns else pd.Series(dtype=float)
        valid = s.notna()
        if valid.sum() < 200 or s[valid].nunique() < 5:
            continue
        q25, q75 = s[valid].quantile([0.25, 0.75]).tolist()
        for side, mask, th in [
            ("LOW_Q", valid & (s <= q25), q25),
            ("HIGH_Q", valid & (s >= q75), q75),
        ]:
            n = int(mask.sum())
            if n < 100:
                continue
            rate = float(y[mask].mean())
            out.append({
                "feature": f, "bucket": side, "threshold": float(th), "rows": n,
                "positive_rate": rate, "base_rate": base,
                "lift": rate / base if base > 0 else None,
            })
    out.sort(key=lambda r: r["lift"] if r["lift"] is not None else -1, reverse=True)
    return out[:30]


def cluster_events(rows: pd.DataFrame, catalog: pd.DataFrame, side: str):
    vectors = []
    for _, c in catalog.iterrows():
        day = c.day
        extreme_bar = int(c.high_bar if side == "HIGH" else c.low_bar)
        g = rows[rows.day == day].sort_values("bar_idx")
        if extreme_bar < 6:
            continue
        w = g[(g.bar_idx >= extreme_bar - 6) & (g.bar_idx <= extreme_bar - 1)].sort_values("bar_idx")
        if len(w) != 6:
            continue
        base_close = float(w.iloc[0].close)
        rec = {
            "day": day, "side": side, "extreme_bar": extreme_bar,
            "extreme_phase": phase(extreme_bar),
            "snapshot_ret_from_open": float(w.iloc[-1].ret_from_open),
            "snapshot_dist_vwap": float(w.iloc[-1].dist_vwap),
            "snapshot_pcb_rel": float(w.iloc[-1].pcb_rel) if pd.notna(w.iloc[-1].pcb_rel) else np.nan,
            "snapshot_breadth": float(w.iloc[-1].pcb_up_breadth) if pd.notna(w.iloc[-1].pcb_up_breadth) else np.nan,
        }
        for j, (_, r) in enumerate(w.iterrows(), start=1):
            rec[f"path_close_{j}"] = float(r.close / base_close - 1.0)
            rec[f"path_vwap_{j}"] = float(r.dist_vwap) if pd.notna(r.dist_vwap) else np.nan
            rec[f"path_pcbrel_{j}"] = float(r.pcb_rel) if pd.notna(r.pcb_rel) else np.nan
        vectors.append(rec)
    ev = pd.DataFrame(vectors)
    if len(ev) < 20:
        return ev, []
    cols = [c for c in ev.columns if c.startswith("path_")]
    X = ev[cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median().fillna(0.0))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(4, max(2, len(ev) // 25))
    km = KMeans(n_clusters=k, n_init=30, random_state=20260903)
    ev["cluster"] = km.fit_predict(Xs)
    summary = []
    for cl, g in ev.groupby("cluster"):
        summary.append({
            "cluster": int(cl), "events": int(len(g)),
            "share": float(len(g) / len(ev)),
            "median_extreme_bar": float(g.extreme_bar.median()),
            "top_phase": str(g.extreme_phase.mode().iloc[0]),
            "median_snapshot_ret_from_open": float(g.snapshot_ret_from_open.median()),
            "median_snapshot_dist_vwap": float(g.snapshot_dist_vwap.median()),
            "median_snapshot_pcb_rel": float(g.snapshot_pcb_rel.median()) if g.snapshot_pcb_rel.notna().any() else None,
            "median_snapshot_breadth": float(g.snapshot_breadth.median()) if g.snapshot_breadth.notna().any() else None,
            "median_path_6bar_return": float(g.path_close_6.median()),
            "median_vwap_change_6bar": float((g.path_vwap_6 - g.path_vwap_1).median()),
            "median_pcbrel_change_6bar": float((g.path_pcbrel_6 - g.path_pcbrel_1).median()) if (g.path_pcbrel_6 - g.path_pcbrel_1).notna().any() else None,
        })
    summary.sort(key=lambda r: r["events"], reverse=True)
    return ev, summary


def time_distribution(catalog: pd.DataFrame):
    out = {}
    for side, col in [("HIGH", "high_phase"), ("LOW", "low_phase")]:
        vc = catalog[col].value_counts().sort_index()
        out[side] = [
            {"phase": str(k), "events": int(v), "share": float(v / len(catalog))}
            for k, v in vc.items()
        ]
    return out


def main():
    rows, catalog = build_frame()
    # Decision rows need a little history. Full 48-bar days remain in catalog.
    decision = rows[(rows.bar_idx >= 3) & (rows.bar_idx <= 45)].copy()

    report = {
        "version": "extreme-pattern-mining-v30",
        "objective": "mechanically inspect every complete 2026 Jan-Aug 5m day to discover repeatable formation patterns before/after the final intraday high/low",
        "analysis_period": "2026-01-01..2026-08-31",
        "reserved_untouched": "2026-09 onward is not used in discovery",
        "days": int(catalog.day.nunique()),
        "future_leakage_policy": "future final high/low is used only as label; predictive features are taken from completed bars at or before decision time",
        "labels": {
            "pre_high_15m": "final daily high occurs 1-3 completed 5m bars ahead",
            "pre_low_15m": "final daily low occurs 1-3 completed 5m bars ahead",
            "post_high_10m": "current bar is 1-2 completed bars after the first final daily high",
            "post_low_10m": "current bar is 1-2 completed bars after the first final daily low",
        },
        "time_distribution": time_distribution(catalog),
        "targets": {},
    }

    for t in TARGETS:
        report["targets"][t] = {
            "positives": int(decision[t].sum()),
            "base_rate": float(decision[t].mean()),
            "oof_shallow_tree": oof_tree_auc(decision, t),
            "interpretable_tree": extract_tree_rules(decision, t),
            "top_univariate_lifts": quantile_lifts(decision, t),
        }

    high_ev, high_clusters = cluster_events(rows, catalog, "HIGH")
    low_ev, low_clusters = cluster_events(rows, catalog, "LOW")
    report["pre_extreme_shape_clusters"] = {"HIGH": high_clusters, "LOW": low_clusters}

    catalog.to_csv(RESULTS / "extreme_catalog_v30.csv", index=False)
    decision.to_csv(RESULTS / "extreme_pattern_rows_v30.csv", index=False)
    pd.concat([high_ev, low_ev], ignore_index=True).to_csv(RESULTS / "extreme_cluster_events_v30.csv", index=False)
    (RESULTS / "extreme_pattern_mining_v30.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTREME PATTERN MINING V3.0")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
