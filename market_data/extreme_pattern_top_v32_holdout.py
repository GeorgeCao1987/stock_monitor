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

DEV_START = pd.Timestamp("2026-01-01").date()
DEV_END = pd.Timestamp("2026-08-31").date()
HOLDOUT_START = pd.Timestamp("2025-03-01").date()
HOLDOUT_END = pd.Timestamp("2025-04-30").date()

TOP_WATCH_THRESHOLD = 0.55
TOP_WATCH_WINDOW = 8  # 40 trading minutes
TOP_GAP_MAX = 0.008
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
    return z, catalog


def fit_full_dev_top(z):
    train = z[
        (z.day >= DEV_START) & (z.day <= DEV_END)
        & z.top_zone.fillna(False)
        & (z.bar_idx >= 3) & (z.bar_idx <= 44)
    ].copy()
    model_med = v31.fit_model(train, "top_forming_15m")
    return model_med, train


def attach_top_probability(z, model_med):
    out = z.copy()
    out["p_top_forming_15m"] = 0.0
    mask = out.top_zone.fillna(False) & (out.bar_idx >= 3) & (out.bar_idx <= 44)
    out.loc[mask, "p_top_forming_15m"] = v31.predict(model_med, out[mask])
    return out


def episode_starts(z, cond):
    q = z.copy().sort_values(["day", "bar_idx"])
    q["_cond"] = pd.Series(cond, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False).astype(bool)
    return q[q._cond & (~prev)].drop(columns="_cond")


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


def frozen_states(z):
    x = add_first_passage(z)
    x["top_watch"] = x.p_top_forming_15m >= TOP_WATCH_THRESHOLD
    x["top_watch_recent"] = x.groupby("day").p_top_forming_15m.transform(
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
    x["reverse_execute_v32"] = (
        (x.bar_idx >= 3) & (x.bar_idx <= MAX_EXEC_BAR)
        & x.top_watch_recent
        & x.recent_high_no_new
        & (x.gap_high <= TOP_GAP_MAX)
        & (x.roll_votes >= 1)
    )
    return x


def watch_metrics(z):
    candidates = z[z.top_zone.fillna(False) & (z.bar_idx >= 3) & (z.bar_idx <= 44)].copy()
    q = episode_starts(candidates, candidates.p_top_forming_15m >= TOP_WATCH_THRESHOLD)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "precision_final_high_within_0_15m": float(q.top_forming_15m.mean()) if len(q) else None,
        "median_probability": float(q.p_top_forming_15m.median()) if len(q) else None,
    }


def execution_metrics(z):
    q = episode_starts(z, z.reverse_execute_v32)
    q = q[q.reverse_execution.notna()].copy()
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "first_passage_win_0_8_vs_0_8": float(q.reverse_execution.mean()) if len(q) else None,
        "median_gap_to_running_high": float(q.gap_high.median()) if len(q) else None,
        "median_roll_votes": float(q.roll_votes.median()) if len(q) else None,
        "events": q[["day", "ts", "close", "p_top_forming_15m", "gap_high", "roll_votes", "reverse_execution"]].to_dict("records"),
    }


def main():
    z, catalog = build_all()
    model_med, train = fit_full_dev_top(z)
    z = attach_top_probability(z, model_med)
    z = frozen_states(z)

    dev = z[(z.day >= DEV_START) & (z.day <= DEV_END)].copy()
    holdout = z[(z.day >= HOLDOUT_START) & (z.day <= HOLDOUT_END)].copy()
    if holdout.day.nunique() < 20:
        raise RuntimeError(f"holdout too small/incomplete: {holdout.day.nunique()} days")

    report = {
        "version": "extreme-pattern-top-v32-holdout",
        "frozen_before_holdout": True,
        "discovery_dev": "2026-01-01..2026-08-31",
        "holdout": "2025-03-01..2025-04-30",
        "holdout_days": int(holdout.day.nunique()),
        "rule": {
            "top_forming_probability_threshold": TOP_WATCH_THRESHOLD,
            "watch_memory_bars": TOP_WATCH_WINDOW,
            "recent_high_no_new": "a new high occurred within the last 3 bars but current bar is not a new high",
            "gap_to_running_high_max": TOP_GAP_MAX,
            "rollover_votes_min": 1,
            "rollover_votes": ["ret1<=0", "dist_vwap_chg1<0", "pcb_rel_chg1<0", "ret3_accel<0", "close_pos_bar<=0.50"],
            "execution_label": "next 12 completed 5m bars: -0.8% favorable move before +0.8% adverse move; same-bar both hit is failure",
        },
        "dev_reference": {
            "watch_oof_v31": {
                "signals": 69,
                "precision": 0.6231884057971014,
                "period_precision": [0.6666666666666666, 0.6428571428571429, 0.55, 0.65],
            },
            "reverse_execute_v32_exploration": {
                "signals": 29,
                "pooled_win": 0.7241379310344828,
                "period_n": [7, 7, 8, 7],
                "period_win": [0.5714285714285714, 0.5714285714285714, 0.75, 1.0],
            },
        },
        "holdout_watch": watch_metrics(holdout),
        "holdout_reverse_execute": execution_metrics(holdout),
        "future_leakage": False,
        "notes": "2025 Mar-Apr was not used to choose the probability threshold, watch memory, gap tolerance, or rollover vote rule.",
    }
    holdout.to_csv(RESULTS / "extreme_pattern_top_v32_holdout_rows.csv", index=False)
    (RESULTS / "extreme_pattern_top_v32_holdout.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("EXTREME PATTERN TOP V3.2 HOLDOUT")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
