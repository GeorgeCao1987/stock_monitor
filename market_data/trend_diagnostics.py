from pathlib import Path
import json
import numpy as np
import pandas as pd

import backtest_v13 as v13
import backtest_v14 as v14
import backtest_v15 as v15
import event_engine_v14 as e14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"


def build():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = v14.add_v14_features(target)
    x = v14.add_context(x)
    x = v14.score_states(x)
    x = v15.add_trend_regime(x)
    x = x.sort_values("ts").reset_index(drop=True)
    # V1.4 state machine uses these columns for structural confirmation.
    x["prev_high"] = x.groupby(x.ts.dt.date).high.shift(1)
    x["prev_low"] = x.groupby(x.ts.dt.date).low.shift(1)
    events = e14.build_events(x)
    ctx_cols = [
        "ts", "trend_state", "up_votes", "down_votes", "pcb_rel_chg3_day",
        "prev_high", "prev_low", "high", "low", "close", "atr_pct",
    ]
    events = events.merge(x[ctx_cols], on="ts", how="left", suffixes=("", "_ctx"))
    events["local_lower_low"] = events.low < events.prev_low
    events["local_higher_high"] = events.high > events.prev_high
    events["pcb_weakening"] = events.pcb_rel_chg3_day < 0
    events["pcb_strengthening"] = events.pcb_rel_chg3_day > 0
    events["top_break_both"] = events.local_lower_low & events.pcb_weakening
    events["bottom_turn_both"] = (~events.local_lower_low) & events.local_higher_high & events.pcb_strengthening
    return x, events


def m(z):
    if z.empty:
        return {"n": 0}
    valid = z.atr_pct.notna() & np.isfinite(z.atr_pct) & (z.atr_pct > 0)
    a = z[valid]
    denom = a.mae_30m.where(a.mae_30m > 1e-9, np.nan)
    return {
        "n": int(len(z)),
        "win_30m_1_5pct": float((z.future_30m >= .015).mean()),
        "win_30m_0_75atr": float((a.mfe_30m >= .75 * a.atr_pct).mean()) if len(a) else None,
        "directional_edge": float((z.mfe_30m > z.mae_30m).mean()),
        "median_mfe": float(z.mfe_30m.median()),
        "median_mae": float(z.mae_30m.median()),
        "median_ratio": float((a.mfe_30m / denom).median()) if len(a) else None,
    }


def section(events, side, event_type):
    z = events[(events.side == side) & (events.event_type == event_type)].copy()
    out = {"ALL": m(z)}
    for state in ["UP", "DOWN", "RANGE"]:
        out[state] = m(z[z.trend_state == state])
    if side == "HIGH":
        out["UP_pcb_weakening"] = m(z[(z.trend_state == "UP") & z.pcb_weakening])
        out["UP_local_lower_low"] = m(z[(z.trend_state == "UP") & z.local_lower_low])
        out["UP_both_at_event"] = m(z[(z.trend_state == "UP") & z.top_break_both])
    else:
        out["DOWN_pcb_strengthening"] = m(z[(z.trend_state == "DOWN") & z.pcb_strengthening])
        out["DOWN_stopped_lower_low"] = m(z[(z.trend_state == "DOWN") & (~z.local_lower_low)])
        out["DOWN_turn_both_at_event"] = m(z[(z.trend_state == "DOWN") & z.bottom_turn_both])
    return out


def main():
    x, events = build()
    report = {
        "trading_days": int(x.ts.dt.date.nunique()),
        "trend_bars": {str(k): int(v) for k, v in x.trend_state.value_counts().to_dict().items()},
        "HIGH_WATCH": section(events, "HIGH", "WATCH_START"),
        "HIGH_CONFIRM": section(events, "HIGH", "STRUCTURE_CONFIRM"),
        "LOW_WATCH": section(events, "LOW", "WATCH_START"),
        "LOW_CONFIRM": section(events, "LOW", "STRUCTURE_CONFIRM"),
        "note": "All groups classify conditions known at each event timestamp; future bars are used only for MFE/MAE scoring.",
    }
    events.to_csv(RESULTS / "trend_diagnostic_events.csv", index=False)
    (RESULTS / "trend_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TREND DIAGNOSTICS")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
