from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
from opening_regime_diagnostics import add_opening_regime
from opening_extreme_forecast_v18 import add_extreme_forecast

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

STAGES = ["WATCH_START", "STRUCTURE_CONFIRM"]


def _event_metrics(z: pd.DataFrame, total_days: int) -> dict:
    if z.empty:
        return {"events": 0, "event_days": 0}
    lag = z.lag_from_daily_extreme
    out = {
        "events": int(len(z)),
        "event_days": int(z.day.nunique()),
        "event_day_coverage": float(z.day.nunique() / total_days) if total_days else None,
        "events_per_event_day": float(len(z) / z.day.nunique()),
        "premature_rate": float((lag < 0).mean()),
        "event_abs_within_1_rate": float((lag.abs() <= 1).mean()),
        "event_abs_within_2_rate": float((lag.abs() <= 2).mean()),
        "event_abs_within_3_rate": float((lag.abs() <= 3).mean()),
        "event_after_within_0_2_rate": float(lag.between(0, 2).mean()),
        "event_after_within_0_3_rate": float(lag.between(0, 3).mean()),
        "median_signed_lag": float(lag.median()),
        "median_abs_lag": float(lag.abs().median()),
    }
    for n in [1, 2, 3]:
        by_day = z.assign(hit=lag.abs() <= n).groupby("day").hit.any()
        out[f"daily_recall_abs_{n}_bars_all_days"] = float(by_day.sum() / total_days) if total_days else None
        out[f"daily_precision_abs_{n}_bars_event_days"] = float(by_day.mean()) if len(by_day) else None
    return out


def build():
    x = e14.build_scored_frame()
    x, daily = add_opening_regime(x)
    daily = add_extreme_forecast(daily)
    events = e14.build_events(x)

    z = events.copy()
    z["day"] = pd.to_datetime(z.ts).dt.date
    ctx = x[["ts", "bar_no"]].drop_duplicates("ts")
    z = z.merge(ctx, on="ts", how="left")
    d = daily[["day", "high_bar_no", "low_bar_no", "forecast"]].copy()
    z = z.merge(d, on="day", how="left")
    z["daily_extreme_bar_no"] = np.where(z.side == "HIGH", z.high_bar_no, z.low_bar_no)
    z["lag_from_daily_extreme"] = z.bar_no - z.daily_extreme_bar_no
    z["opening_forecast_at_event"] = np.where(z.bar_no >= 5, z.forecast, "PENDING")
    return x, daily, z


def main():
    x, daily, events = build()
    total_days = int(len(daily))
    report = {
        "version": "stage-extreme-replay-v19",
        "objective": "Replay every mechanically generated event and compare WATCH vs STRUCTURE_CONFIRM distance to the true daily extreme.",
        "future_labels_only_for_scoring": True,
        "candidate_generation_changed": False,
        "trading_days": total_days,
        "by_side_stage": {},
        "role_split": {
            "HIGH_PRIMARY": "WATCH_START",
            "LOW_PRIMARY": "STRUCTURE_CONFIRM",
            "status": "diagnostic_only_until_holdout_validation",
        },
    }
    for side in ["HIGH", "LOW"]:
        report["by_side_stage"][side] = {}
        for stage in STAGES:
            q = events[(events.side == side) & (events.event_type == stage)].copy()
            report["by_side_stage"][side][stage] = _event_metrics(q, total_days)

    # Forecast-conditioned view is intentionally descriptive; no filter is fitted here.
    report["by_forecast"] = {}
    for side, stage in [("HIGH", "WATCH_START"), ("LOW", "STRUCTURE_CONFIRM")]:
        q = events[(events.side == side) & (events.event_type == stage) & (events.bar_no >= 5)].copy()
        report["by_forecast"][f"{side}_{stage}"] = {
            state: _event_metrics(q[q.forecast == state], total_days)
            for state in ["HIGH_AHEAD", "LOW_AHEAD", "UNCERTAIN"]
        }

    events.to_csv(RESULTS / "stage_extreme_events_v19.csv", index=False)
    (RESULTS / "stage_extreme_replay_v19.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("STAGE EXTREME REPLAY V1.9")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
