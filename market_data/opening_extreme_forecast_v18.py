from pathlib import Path
import json
import pandas as pd

import event_engine_v14 as e14
from opening_regime_diagnostics import add_opening_regime, OPEN_BARS

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# Frozen after cross-period development analysis on 2026-01..08.
# HIGH is intentionally stricter than LOW.  The objective is not final close
# direction; it is whether the corresponding daily extreme is still ahead
# after the 10:00 snapshot.
HIGH_UP_VOTES = 4
HIGH_NET_MARGIN = 2
LOW_DOWN_VOTES = 2
LOW_NET_MARGIN = 0


def forecast_from_votes(up_votes: int, down_votes: int) -> str:
    if up_votes >= HIGH_UP_VOTES and up_votes - down_votes >= HIGH_NET_MARGIN:
        return "HIGH_AHEAD"
    if down_votes >= LOW_DOWN_VOTES and down_votes - up_votes >= LOW_NET_MARGIN:
        return "LOW_AHEAD"
    return "UNCERTAIN"


def add_extreme_forecast(daily: pd.DataFrame) -> pd.DataFrame:
    z = daily.copy()
    z["forecast"] = [
        forecast_from_votes(int(u), int(d))
        for u, d in zip(z.opening_up_votes, z.opening_down_votes)
    ]
    z["high_after_10"] = z.high_bar_no >= OPEN_BARS
    z["low_after_10"] = z.low_bar_no >= OPEN_BARS
    z["high_ahead_pattern"] = z.high_after_10 & (~z.low_after_10)
    z["low_ahead_pattern"] = (~z.high_after_10) & z.low_after_10
    return z


def state_metrics(z: pd.DataFrame) -> dict:
    if z.empty:
        return {"days": 0}
    return {
        "days": int(len(z)),
        "high_after_10_rate": float(z.high_after_10.mean()),
        "low_after_10_rate": float(z.low_after_10.mean()),
        "high_ahead_pattern_rate": float(z.high_ahead_pattern.mean()),
        "low_ahead_pattern_rate": float(z.low_ahead_pattern.mean()),
        "median_high_bar": float(z.high_bar_no.median()),
        "median_low_bar": float(z.low_bar_no.median()),
    }


def main():
    x = e14.build_scored_frame()
    _, daily = add_opening_regime(x)
    daily = add_extreme_forecast(daily)

    states = {
        state: state_metrics(daily[daily.forecast == state])
        for state in ["HIGH_AHEAD", "LOW_AHEAD", "UNCERTAIN"]
    }
    report = {
        "version": "opening-extreme-forecast-v18",
        "objective": "At 10:00 predict which side of the daily extreme is still likely ahead, not the final close direction.",
        "frozen_rules": {
            "HIGH_AHEAD": f"up_votes >= {HIGH_UP_VOTES} and up_votes-down_votes >= {HIGH_NET_MARGIN}",
            "LOW_AHEAD": f"down_votes >= {LOW_DOWN_VOTES} and down_votes-up_votes >= {LOW_NET_MARGIN}",
            "UNCERTAIN": "otherwise",
            "candidate_generation_changed": False,
            "v17_turn_confirmation_changed": False,
            "no_future_leakage": True,
        },
        "trading_days": int(len(daily)),
        "forecast_counts": daily.forecast.value_counts().to_dict(),
        "states": states,
        "headline": {
            "HIGH_AHEAD_precision": states["HIGH_AHEAD"].get("high_after_10_rate"),
            "LOW_AHEAD_precision": states["LOW_AHEAD"].get("low_after_10_rate"),
        },
        "note": (
            "Daily high/low locations are future labels used only for evaluation. "
            "The forecast itself uses the completed opening snapshot only."
        ),
    }
    daily.to_csv(RESULTS / "opening_extreme_forecast_v18.csv", index=False)
    (RESULTS / "opening_extreme_forecast_v18.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("OPENING EXTREME FORECAST V1.8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
