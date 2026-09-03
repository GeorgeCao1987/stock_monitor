from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import t_edge_daily_regime_model_v29 as v29
import extreme_pattern_mining_v30 as v30
import extreme_pattern_state_v31 as v31

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

TOP_WATCH_THRESHOLD = 0.55
TOP_WATCH_WINDOW = 8
TOP_GAP_MAX_V32 = 0.008
TOP_GAP_ATR_MAX_V33 = 0.75
MAX_EXEC_BAR = 35
EXEC_TARGET = 0.008
EXEC_STOP = 0.008
EXEC_BARS = 12


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_all():
    x = e14.build_scored_frame()
    x = v30.add_past_features(x)
    d = v29.daily_context()
    x = x.merge(d, on="day", how="left")
    rows, catalog = v30.add_extreme_labels(x)
    z = rows.sort_values(["day", "bar_idx"]).reset_index(drop=True)
    z["top_forming_15m"] = ((z.bars_to_final_high >= 0) & (z.bars_to_final_high <= 3)).astype(int)
    z["top_zone"] = (z.pos_in_range >= 0.72) | z.new_high.fillna(False)
    z["gap_high"] = ((z.cum_high - z.close) / z.close).clip(lower=0)
    z["gap_high_atr"] = z.gap_high / _num(z.atr_pct).replace(0, np.nan)
    return z, catalog


def candidate_mask(z):
    return z.top_zone.fillna(False) & (z.bar_idx >= 3) & (z.bar_idx <= 44)


def add_oof_top_probability(z):
    out = z.copy()
    out["p_top_forming_15m"] = np.nan
    fold_report = {}
    dev = out[(out.day >= DEV_START) & (out.day <= DEV_END)].copy()
    for name, start, end in PERIODS:
        test_mask = (out.day >= start) & (out.day <= end) & candidate_mask(out)
        train = dev[~((dev.day >= start) & (dev.day <= end)) & candidate_mask(dev)].copy()
        test = out[test_mask].copy()
        if len(test) == 0 or len(train) == 0:
            raise RuntimeError(f"missing OOF data for {name}")
        model_med = v31.fit_model(train, "top_forming_15m")
        p = v31.predict(model_med, test)
        out.loc[test.index, "p_top_forming_15m"] = p
        y = test.top_forming_15m.astype(int)
        from sklearn.metrics import roc_auc_score, brier_score_loss
        fold_report[name] = {
            "candidate_rows": int(len(test)),
            "positives": int(y.sum()),
            "base_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
            "brier": float(brier_score_loss(y, p)),
        }
    return out, fold_report


def fit_full_dev_top(z):
    train = z[(z.day >= DEV_START) & (z.day <= DEV_END) & candidate_mask(z)].copy()
    return v31.fit_model(train, "top_forming_15m")


def attach_holdout_probability(z, model_med):
    out = z.copy()
    out["p_top_forming_15m"] = np.nan
    mask = (out.day >= HOLDOUT_START) & (out.day <= HOLDOUT_END) & candidate_mask(out)
    if mask.sum():
        out.loc[mask, "p_top_forming_15m"] = v31.predict(model_med, out[mask])
    return out


def add_first_passage(z):
    x = z.copy().sort_values(["day", "bar_idx"]).reset_index(drop=True)
    rev = []
    for i, r in x.iterrows():
        if int(r.bar_idx) > MAX_EXEC_BAR:
            rev.append(np.nan)
            continue
        fut = x.iloc[i + 1:i + 1 + EXEC_BARS]
        fut = fut[fut.day == r.day]
        if len(fut) < EXEC_BARS:
            rev.append(np.nan)
            continue
        label = 0
        for _, f in fut.iterrows():
            down = float(r.close / f.low - 1.0)
            up = float(f.high / r.close - 1.0)
            if down >= EXEC_TARGET or up >= EXEC_STOP:
                label = int(down >= EXEC_TARGET and up < EXEC_STOP)
                break
        rev.append(label)
    x["reverse_execution"] = rev
    return x


def add_states(z):
    x = add_first_passage(z)
    p = _num(x.p_top_forming_15m).fillna(0.0)
    x["top_watch"] = p >= TOP_WATCH_THRESHOLD
    x["top_watch_recent"] = p.groupby(x.day).transform(
        lambda s: s.rolling(TOP_WATCH_WINDOW, min_periods=1).max()
    ) >= TOP_WATCH_THRESHOLD
    x["recent_high_no_new"] = (x.new_high_count3 >= 1) & (~x.new_high.fillna(False))
    x["roll_votes"] = (
        (_num(x.ret1) <= 0).astype(int)
        + (_num(x.dist_vwap_chg1) < 0).astype(int)
        + (_num(x.pcb_rel_chg1) < 0).astype(int)
        + (_num(x.ret3_accel) < 0).astype(int)
        + (_num(x.close_pos_bar) <= 0.50).astype(int)
    )
    core = (
        (x.bar_idx >= 3) & (x.bar_idx <= MAX_EXEC_BAR)
        & x.top_watch_recent
        & x.recent_high_no_new
        & (x.roll_votes >= 1)
    )
    x["reverse_execute_v32"] = core & (x.gap_high <= TOP_GAP_MAX_V32)
    # V3.3 only changes tradeability: current price must still be close to the
    # running high in both absolute terms and relative to signal-time 5m ATR.
    x["near_high_adaptive_v33"] = (
        (x.gap_high <= TOP_GAP_MAX_V32)
        & x.gap_high_atr.notna()
        & (x.gap_high_atr <= TOP_GAP_ATR_MAX_V33)
    )
    x["reverse_execute_v33"] = core & x.near_high_adaptive_v33
    return x


def episode_starts(z, cond):
    q = z.copy().sort_values(["day", "bar_idx"])
    q["_cond"] = pd.Series(cond, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False).astype(bool)
    return q[q._cond & (~prev)].drop(columns="_cond")


def watch_metrics(z):
    c = z[candidate_mask(z)].copy()
    q = episode_starts(c, _num(c.p_top_forming_15m) >= TOP_WATCH_THRESHOLD)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "precision_final_high_within_0_15m": float(q.top_forming_15m.mean()) if len(q) else None,
        "median_probability": float(q.p_top_forming_15m.median()) if len(q) else None,
    }


def execution_metrics(z, col):
    q = episode_starts(z, z[col].fillna(False))
    q = q[q.reverse_execution.notna()].copy()
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "first_passage_win_0_8_vs_0_8": float(q.reverse_execution.mean()) if len(q) else None,
        "median_gap_high": float(q.gap_high.median()) if len(q) else None,
        "median_gap_high_atr": float(q.gap_high_atr.median()) if len(q) else None,
        "events": q[["day", "ts", "close", "p_top_forming_15m", "gap_high", "gap_high_atr", "roll_votes", "reverse_execution"]].to_dict("records"),
    }


def period_metrics(z, col):
    out = {}
    for name, start, end in PERIODS:
        q = z[(z.day >= start) & (z.day <= end)].copy()
        out[name] = execution_metrics(q, col)
    return out


def development_pass(v32, v33, v33_periods):
    valid3 = [m for m in v33_periods.values() if m["signals"] >= 3 and m["first_passage_win_0_8_vs_0_8"] is not None]
    valid2 = [m for m in v33_periods.values() if m["signals"] >= 2]
    min_win = min((m["first_passage_win_0_8_vs_0_8"] for m in valid3), default=None)
    return bool(
        v33["signals"] >= 25
        and len(valid2) >= 6
        and min_win is not None and min_win >= 0.50
        and v33["first_passage_win_0_8_vs_0_8"] is not None
        and v33["first_passage_win_0_8_vs_0_8"] >= 0.70
        and v32["signals"] > 0
        and v33["signals"] >= 0.50 * v32["signals"]
        and v32["first_passage_win_0_8_vs_0_8"] is not None
        and v33["first_passage_win_0_8_vs_0_8"] >= v32["first_passage_win_0_8_vs_0_8"]
    )


def main():
    raw, catalog = build_all()
    dev_oof, fold_report = add_oof_top_probability(raw)
    dev_oof = add_states(dev_oof)
    dev = dev_oof[(dev_oof.day >= DEV_START) & (dev_oof.day <= DEV_END)].copy()

    dev_v32 = execution_metrics(dev, "reverse_execute_v32")
    dev_v33 = execution_metrics(dev, "reverse_execute_v33")
    dev_v32_periods = period_metrics(dev, "reverse_execute_v32")
    dev_v33_periods = period_metrics(dev, "reverse_execute_v33")
    dev_watch = watch_metrics(dev)
    passed = development_pass(dev_v32, dev_v33, dev_v33_periods)

    holdout_report = {"evaluated": False, "reason": "development gate failed; holdout preserved"}
    holdout_rows = pd.DataFrame()
    if passed:
        model_med = fit_full_dev_top(raw)
        h = raw[(raw.day >= HOLDOUT_START) & (raw.day <= HOLDOUT_END)].copy()
        h = attach_holdout_probability(h, model_med)
        h = add_states(h)
        if h.day.nunique() < 20:
            raise RuntimeError(f"holdout too small/incomplete: {h.day.nunique()} days")
        holdout_report = {
            "evaluated": True,
            "days": int(h.day.nunique()),
            "watch": watch_metrics(h),
            "v32_baseline": execution_metrics(h, "reverse_execute_v32"),
            "v33_adaptive": execution_metrics(h, "reverse_execute_v33"),
        }
        holdout_rows = h

    report = {
        "version": "extreme-pattern-adaptive-v33",
        "optimization": "keep V3.2 top-formation model/state machine; add adaptive near-high tradeability gate gap_high <= 0.75 x signal-time 5m ATR while retaining absolute <=0.8% cap",
        "development": "2025-03..2026-08, nine already-consumed double-month periods, leave-one-period-out formation probabilities",
        "reserved_holdout": "2025-01..02; evaluated only if development gate passes",
        "rules_frozen_before_holdout": True,
        "parameters": {
            "top_watch_threshold": TOP_WATCH_THRESHOLD,
            "watch_memory_bars": TOP_WATCH_WINDOW,
            "v32_gap_max": TOP_GAP_MAX_V32,
            "v33_gap_atr_max": TOP_GAP_ATR_MAX_V33,
            "roll_votes_min": 1,
            "execution_target": EXEC_TARGET,
            "execution_stop": EXEC_STOP,
            "execution_bars": EXEC_BARS,
        },
        "formation_fold_metrics": fold_report,
        "development_watch": dev_watch,
        "development_v32_baseline": {"pooled": dev_v32, "periods": dev_v32_periods},
        "development_v33_adaptive": {"pooled": dev_v33, "periods": dev_v33_periods},
        "development_gate": {
            "passed": passed,
            "policy": "V3.3 >=25 signals; >=6/9 periods with >=2 signals; worst among periods with >=3 signals >=50%; pooled >=70%; retain >=50% of V3.2 signals; pooled win >= V3.2",
        },
        "holdout": holdout_report,
        "bottom_side": "no executable positive-T rule promoted; V3.1/V3.2 bottom confirmation remained unstable and is kept OBSERVE_ONLY",
        "future_leakage": False,
    }
    dev.to_csv(RESULTS / "extreme_pattern_adaptive_v33_dev_rows.csv", index=False)
    if len(holdout_rows):
        holdout_rows.to_csv(RESULTS / "extreme_pattern_adaptive_v33_holdout_rows.csv", index=False)
    (RESULTS / "extreme_pattern_adaptive_v33.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("EXTREME PATTERN ADAPTIVE V3.3")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
