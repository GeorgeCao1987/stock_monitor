from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import event_engine_v17 as e17
from opening_regime_diagnostics import add_opening_regime
from opening_extreme_forecast_v18 import add_extreme_forecast

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def _metrics(z: pd.DataFrame) -> dict:
    if z.empty:
        return {"n": 0}
    lag = z.lag_from_daily_extreme
    nonneg = lag[lag >= 0]
    return {
        "n": int(len(z)),
        "premature_rate": float((lag < 0).mean()),
        "at_or_after_extreme_rate": float((lag >= 0).mean()),
        "within_0_1_bars_after_rate": float(lag.between(0, 1).mean()),
        "within_0_2_bars_after_rate": float(lag.between(0, 2).mean()),
        "within_0_3_bars_after_rate": float(lag.between(0, 3).mean()),
        "median_lag_bars_if_not_premature": float(nonneg.median()) if len(nonneg) else None,
        "directional_30m_rate": float((z.mfe_30m > z.mae_30m).mean()),
    }


def main():
    x = e14.build_scored_frame()
    x, daily = add_opening_regime(x)
    daily = add_extreme_forecast(daily)
    _, actionable = e17.build_v17(x)

    a = actionable.copy()
    a["day"] = pd.to_datetime(a.ts).dt.date
    d = daily[["day", "high_bar_no", "low_bar_no", "forecast"]].copy()
    a = a.merge(d, on="day", how="left")
    a["daily_extreme_bar_no"] = np.where(a.side == "HIGH", a.high_bar_no, a.low_bar_no)
    a["lag_from_daily_extreme"] = a.bar_no - a.daily_extreme_bar_no
    a["opening_forecast_available"] = a.bar_no >= 5  # completed 10:00 bar

    report = {
        "version": "extreme-capture-diagnostics-v18",
        "objective": "Measure how close V1.7 actionable confirmations are to the true daily high/low in 5-minute bars.",
        "future_label_only": True,
        "overall": {},
        "post_opening_by_forecast": {},
    }

    for side in ["HIGH", "LOW"]:
        z = a[a.side == side]
        report["overall"][side] = _metrics(z)
        zp = z[z.opening_forecast_available]
        report["post_opening_by_forecast"][side] = {
            state: _metrics(zp[zp.forecast == state])
            for state in ["HIGH_AHEAD", "LOW_AHEAD", "UNCERTAIN"]
        }

    a.to_csv(RESULTS / "extreme_capture_actions_v18.csv", index=False)
    (RESULTS / "extreme_capture_diagnostics_v18.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTREME CAPTURE DIAGNOSTICS V1.8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
