from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as v14e

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def bar_number_map(x):
    z = x[["ts"]].copy()
    z["day"] = z.ts.dt.date
    z["bar_no"] = z.groupby("day").cumcount()
    return z.set_index("ts")["bar_no"].to_dict()


def metrics(a):
    if a.empty:
        return {
            "events": 0,
            "win_15m_1pct": None,
            "win_30m_1_5pct": None,
            "win_30m_0_75atr": None,
            "win_30m_1atr": None,
            "directional_edge_mfe_gt_mae": None,
            "median_mfe_30m": None,
            "median_mae_30m": None,
            "median_mfe_mae_ratio": None,
        }
    z = a.copy()
    valid = z.atr_pct.notna() & np.isfinite(z.atr_pct) & (z.atr_pct > 0)
    za = z.loc[valid].copy()
    if not za.empty:
        denom = za.mae_30m.where(za.mae_30m > 1e-9, np.nan)
        za["mfe_mae_ratio"] = za.mfe_30m / denom
    return {
        "events": int(len(z)),
        "win_15m_1pct": float((z.future_15m >= .01).mean()),
        "win_30m_1_5pct": float((z.future_30m >= .015).mean()),
        "win_30m_0_75atr": float((za.mfe_30m >= .75 * za.atr_pct).mean()) if len(za) else None,
        "win_30m_1atr": float((za.mfe_30m >= za.atr_pct).mean()) if len(za) else None,
        "directional_edge_mfe_gt_mae": float((z.mfe_30m > z.mae_30m).mean()),
        "median_mfe_30m": float(z.mfe_30m.median()),
        "median_mae_30m": float(z.mae_30m.median()),
        "median_mfe_mae_ratio": float(za.mfe_mae_ratio.median()) if len(za) else None,
    }


def build_v17(x):
    # V1.4 state machine generates events using information available at each bar.
    # V1.7 only changes when an event becomes actionable.
    ev = v14e.build_events(x).copy()
    if ev.empty:
        return ev, ev

    bmap = bar_number_map(x)
    ev["bar_no"] = ev.ts.map(bmap)
    ev["start_bar_no"] = ev.start_ts.map(bmap)
    ev["confirm_lag_bars"] = ev.bar_no - ev.start_bar_no

    # High side: WATCH is observation only. Execute only when structure confirmation
    # arrives on the 1st or 2nd subsequent 5-minute trading bar. Scoring starts at
    # the confirmation timestamp itself, so no future information enters the action.
    high_quick = ev[
        (ev.side == "HIGH") &
        (ev.event_type == "STRUCTURE_CONFIRM") &
        (ev.confirm_lag_bars >= 1) &
        (ev.confirm_lag_bars <= 2)
    ].copy()
    high_quick["v17_action"] = "HIGH_QUICK_CONFIRM_EXECUTE"

    # Low side: retain V1.4 right-side structure confirmation unchanged.
    low_confirm = ev[
        (ev.side == "LOW") &
        (ev.event_type == "STRUCTURE_CONFIRM")
    ].copy()
    low_confirm["v17_action"] = "LOW_STRUCTURE_CONFIRM_EXECUTE"

    actionable = pd.concat([high_quick, low_confirm], ignore_index=True).sort_values("ts")
    return ev, actionable


def summarize(x, ev, actionable):
    high_quick = actionable[actionable.side == "HIGH"]
    low_confirm = actionable[actionable.side == "LOW"]
    high_watch = ev[(ev.side == "HIGH") & (ev.event_type == "WATCH_START")]
    high_late = ev[
        (ev.side == "HIGH") &
        (ev.event_type == "STRUCTURE_CONFIRM") &
        (ev.confirm_lag_bars >= 3)
    ]
    days = int(x.ts.dt.date.nunique())
    return {
        "version": "V1.7-Core",
        "policy": {
            "high_watch": "observe_only",
            "high_execute": "structure confirm on bar +1 or +2; action timestamp is confirmation bar",
            "high_late_confirm": "stale_no_action",
            "low_execute": "retain V1.4 structure confirm",
            "no_future_leakage": True,
        },
        "trading_days": days,
        "HIGH_WATCH_REFERENCE": metrics(high_watch),
        "HIGH_QUICK_CONFIRM_ACTION": metrics(high_quick),
        "HIGH_LATE_CONFIRM_REFERENCE": metrics(high_late),
        "LOW_CONFIRM_ACTION": metrics(low_confirm),
        "ACTIONABLE": {
            **metrics(actionable),
            "events_per_day": float(len(actionable) / days) if days else None,
        },
    }


def main():
    x = v14e.build_scored_frame()
    ev, actionable = build_v17(x)
    actionable.to_csv(RESULTS / "events_v17.csv", index=False)
    summary = summarize(x, ev, actionable)
    (RESULTS / "summary_events_v17.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V1.7 EVENT SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
