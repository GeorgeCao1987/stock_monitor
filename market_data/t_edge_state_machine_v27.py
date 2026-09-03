from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import rolling_extreme_probability_v23 as v23

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET_MOVE = 0.008
STOP_MOVE = 0.008
FP_HORIZON_BARS = 12  # 60 trading minutes
POS_CONFIRM_MAX_WAIT = 9  # 45 trading minutes after WATCH


def add_first_passage_labels(z: pd.DataFrame) -> pd.DataFrame:
    """Scoring-only execution label.

    POSITIVE succeeds if +0.8% is reached before -0.8% within the next 60
    trading minutes. REVERSE is symmetric. If both barriers are touched in the
    same 5m bar, count conservatively as failure because intrabar order is
    unknown. Future bars never enter signal generation.
    """
    x = z.copy().sort_values(["day", "ts"]).reset_index(drop=True)
    pos, rev = [], []
    for i, r in x.iterrows():
        future = x.iloc[i + 1:i + 1 + FP_HORIZON_BARS]
        future = future[future.day == r.day]

        p = 0
        for _, f in future.iterrows():
            up = float(f.high / r.close - 1.0)
            down = float(r.close / f.low - 1.0)
            if up >= TARGET_MOVE or down >= STOP_MOVE:
                p = int(up >= TARGET_MOVE and down < STOP_MOVE)
                break
        pos.append(p)

        q = 0
        for _, f in future.iterrows():
            down = float(r.close / f.low - 1.0)
            up = float(f.high / r.close - 1.0)
            if down >= TARGET_MOVE or up >= STOP_MOVE:
                q = int(down >= TARGET_MOVE and up < STOP_MOVE)
                break
        rev.append(q)

    x["positive_first_passage_win"] = pos
    x["reverse_first_passage_win"] = rev
    return x


def base_state() -> pd.DataFrame:
    x = e14.build_scored_frame()
    z = v23.add_v23_state(x)
    z = z[z.eligible_realtime].copy()
    z = add_first_passage_labels(z)
    z["positive_structural_edge"] = (
        (z.remaining_upside_from_close >= TARGET_MOVE)
        & (z.remaining_upside_from_close > z.remaining_downside_from_close)
    )
    z["reverse_structural_edge"] = (
        (z.remaining_downside_from_close >= TARGET_MOVE)
        & (z.remaining_downside_from_close > z.remaining_upside_from_close)
    )
    return z


def positive_watch_mask(z: pd.DataFrame) -> pd.Series:
    return (
        z.time_phase.isin(["EARLY_0950_1030", "LATE_AM_1035_1130"])
        & z.bottom_tradeable_now.fillna(False)
        & (pd.to_numeric(z.ret_from_open, errors="coerce") <= -0.03)
        & (pd.to_numeric(z.dist_vwap, errors="coerce") <= -0.005)
    )


def reverse_execute_mask(z: pd.DataFrame) -> pd.Series:
    roll_count = (
        (pd.to_numeric(z.ret1, errors="coerce") < 0).astype(int)
        + (pd.to_numeric(z.dist_vwap_chg1, errors="coerce") < 0).astype(int)
        + (pd.to_numeric(z.pcb_rel_chg3, errors="coerce") < 0).astype(int)
    )
    return (
        (z.time_phase == "EARLY_0950_1030")
        & z.top_tradeable_now.fillna(False)
        & (pd.to_numeric(z.ret_from_open, errors="coerce") >= 0.01)
        & (pd.to_numeric(z.dist_vwap, errors="coerce") >= 0.005)
        & (roll_count >= 2)
    )


def episode_starts(z: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    q = z.copy().sort_values(["day", "ts"])
    q["_mask"] = pd.Series(mask, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_mask"].shift(1).fillna(False)
    return q[q._mask & (~prev)].drop(columns="_mask")


def positive_execute_events(z: pd.DataFrame) -> pd.DataFrame:
    """LOW-specific two-stage execution: WATCH first, then rejection confirm.

    Confirmation is `failed_breakdown`: the current bar makes a new running low
    but closes back above the prior running low. It must occur within 9 bars of
    the first WATCH and while price remains bottom-tradeable.
    """
    rows = []
    watch = positive_watch_mask(z)
    for day, g0 in z.groupby("day", sort=True):
        g = g0.sort_values("ts")
        watch_indices = list(g.index[watch.reindex(g.index).fillna(False)])
        if not watch_indices:
            continue
        wi = watch_indices[0]
        idxs = list(g.index)
        pos = idxs.index(wi)
        for step, i in enumerate(idxs[pos + 1:pos + 1 + POS_CONFIRM_MAX_WAIT], start=1):
            r = z.loc[i]
            if r.time_phase not in ["EARLY_0950_1030", "LATE_AM_1035_1130"]:
                break
            if bool(r.failed_breakdown) and bool(r.bottom_tradeable_now):
                rec = r.to_dict()
                rec["watch_ts"] = z.loc[wi, "ts"]
                rec["wait_bars"] = step
                rows.append(rec)
                break
    return pd.DataFrame(rows)


def metrics(q: pd.DataFrame, side: str) -> dict:
    if q.empty:
        return {"signals": 0, "signal_days": 0}
    if side == "POSITIVE":
        fp = "positive_first_passage_win"
        structural = "positive_structural_edge"
        favorable = "remaining_upside_from_close"
        wrong = "remaining_downside_from_close"
    else:
        fp = "reverse_first_passage_win"
        structural = "reverse_structural_edge"
        favorable = "remaining_downside_from_close"
        wrong = "remaining_upside_from_close"
    ratio = q[favorable] / q[wrong].replace(0, np.nan)
    return {
        "signals": int(len(q)),
        "signal_days": int(pd.Series(q.day).nunique()),
        "first_passage_win_rate": float(q[fp].mean()),
        "structural_edge_rate": float(q[structural].mean()),
        "move_hit_rate": float((q[favorable] >= TARGET_MOVE).mean()),
        "median_favorable_room": float(q[favorable].median()),
        "median_wrong_way_room": float(q[wrong].median()),
        "median_room_ratio": float(ratio.replace([np.inf, -np.inf], np.nan).median()) if ratio.notna().any() else None,
    }


def summarize_period(z: pd.DataFrame) -> dict:
    pw = episode_starts(z, positive_watch_mask(z))
    pe = positive_execute_events(z)
    re = episode_starts(z, reverse_execute_mask(z))
    out = {
        "POSITIVE_WATCH": metrics(pw, "POSITIVE"),
        "POSITIVE_EXECUTE": metrics(pe, "POSITIVE"),
        "REVERSE_EXECUTE": metrics(re, "REVERSE"),
    }
    if not pe.empty:
        out["POSITIVE_EXECUTE"]["median_wait_bars"] = float(pe.wait_bars.median())
    return out


def main():
    z = base_state()
    periods = [
        ("JAN_FEB", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-28").date()),
        ("MAR_APR", pd.Timestamp("2026-03-01").date(), pd.Timestamp("2026-04-30").date()),
        ("MAY_JUN", pd.Timestamp("2026-05-01").date(), pd.Timestamp("2026-06-30").date()),
    ]
    per = {}
    for name, start, end in periods:
        q = z[(z.day >= start) & (z.day <= end)].copy()
        if not q.empty:
            per[name] = summarize_period(q)

    report = {
        "version": "t-edge-state-machine-v27",
        "status": "frozen_development_candidate_before_new_holdout",
        "primary_live_semantics": {
            "POSITIVE_WATCH": "deep morning selloff has a historically favorable later rebound structure; observe only",
            "POSITIVE_EXECUTE": "after WATCH, failed-breakdown rejection confirms positive-T entry",
            "REVERSE_EXECUTE": "early overbought/tradeable top plus at least two rollover signals; reverse-T can act without waiting for LOW-style confirmation",
        },
        "execution_label": "+/-0.8% first-passage over next 60 trading minutes; same-bar both barriers = conservative failure",
        "positive_watch_rule": "morning + bottom_tradeable + ret_from_open<=-3% + dist_vwap<=-0.5%",
        "positive_execute_rule": "first failed_breakdown within 9 bars after first WATCH, still morning and bottom_tradeable",
        "reverse_execute_rule": "EARLY_0950_1030 + top_tradeable + ret_from_open>=+1% + dist_vwap>=+0.5% + at least 2 of ret1<0, dist_vwap_chg1<0, pcb_rel_chg3<0",
        "future_leakage": False,
        "periods": per,
        "note": "Rules are frozen on 2026 Jan-Jun development before the new 2025 May-Jun holdout is inspected.",
    }
    (RESULTS / "t_edge_state_machine_v27.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    z.to_csv(RESULTS / "t_edge_state_machine_v27_rows.csv", index=False)
    print("T EDGE STATE MACHINE V2.7")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
