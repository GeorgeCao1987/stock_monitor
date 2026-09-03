from pathlib import Path
import json
import pandas as pd

import backtest_v13 as v13
import backtest_v14 as v14
import backtest_v15 as v15
import event_engine_v14 as e14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"
GRACE_FALSE_BARS = e14.GRACE_FALSE_BARS


def build_scored_frame():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = v14.add_v14_features(target)
    x = v14.add_context(x)
    x = v15.score_states(x)
    x = x.sort_values("ts").reset_index(drop=True)
    x["prev_high"] = x.groupby(x.ts.dt.date).high.shift(1)
    x["prev_low"] = x.groupby(x.ts.dt.date).low.shift(1)
    return x


def emit_event(events, x, i, side, event_id, event_type, action, start_ts):
    r = x.loc[i]
    score = r.high_score if side == "HIGH" else r.low_score
    rec = {
        "event_id": event_id,
        "ts": r.ts,
        "start_ts": start_ts,
        "side": side,
        "event_type": event_type,
        "action": action,
        "price": r.close,
        "score": score,
        "trend_state": r.trend_state,
        "up_votes": r.up_votes,
        "down_votes": r.down_votes,
        "ret_from_open": r.ret_from_open,
        "dist_vwap": r.dist_vwap,
        "pos_in_range": r.pos_in_range,
        "vol_ratio": r.vol_ratio,
        "atr_pct": r.atr_pct,
        "pcb_rel": r.pcb_rel,
        "pcb_rel_chg3": r.pcb_rel_chg3,
        "pcb_rel_chg3_day": r.pcb_rel_chg3_day,
        "uptrend_top_break": bool(r.uptrend_top_break),
        "downtrend_bottom_turn": bool(r.downtrend_bottom_turn),
    }
    rec.update(v13.evaluate_future(x, i, side))
    events.append(rec)


def build_events(x):
    events = []
    seq = 0
    for side in ["HIGH", "LOW"]:
        watch_col = "high_watch" if side == "HIGH" else "low_watch"
        for day, g in x.groupby(x.ts.dt.date, sort=True):
            active = False
            false_streak = 0
            confirmed = False
            start_i = None
            event_id = None
            for i in g.index:
                watch = bool(x.loc[i, watch_col])
                if not active:
                    if not watch:
                        continue
                    seq += 1
                    event_id = f"{side}-{day}-{seq}"
                    active = True
                    false_streak = 0
                    confirmed = False
                    start_i = i
                    action = "REDUCE_OR_REVERSE_T_WATCH" if side == "HIGH" else "OBSERVE_ONLY"
                    emit_event(events, x, i, side, event_id, "WATCH_START", action, x.loc[i, "ts"])
                    continue

                if watch:
                    false_streak = 0
                else:
                    false_streak += 1

                if not confirmed and i > start_i:
                    if side == "HIGH":
                        confirm = pd.notna(x.loc[i, "prev_low"]) and x.loc[i, "close"] < x.loc[i, "prev_low"]
                        action = "TOP_STRUCTURE_CONFIRM"
                    else:
                        confirm = pd.notna(x.loc[i, "prev_high"]) and x.loc[i, "close"] > x.loc[i, "prev_high"]
                        action = "LOW_ENTRY_CONFIRM"
                    if confirm:
                        confirmed = True
                        emit_event(events, x, i, side, event_id, "STRUCTURE_CONFIRM", action, x.loc[start_i, "ts"])

                if false_streak >= GRACE_FALSE_BARS:
                    active = False
                    false_streak = 0
                    confirmed = False
                    start_i = None
                    event_id = None

    return pd.DataFrame(events).sort_values("ts").reset_index(drop=True)


def metrics_by_trend(events):
    out = {}
    if events.empty or "trend_state" not in events.columns:
        return out
    for state in ["UP", "DOWN", "RANGE"]:
        z = events[events.trend_state == state]
        if not z.empty:
            out[state] = e14.metrics(z)
    return out


def summarize(x, events):
    base = e14.summarize(x, events)
    base["model"] = "V1.5"
    base["change"] = "V1.4 scores frozen; trend-regime gating before event state machine"
    high_watch = events[(events.side == "HIGH") & (events.event_type == "WATCH_START")]
    low_confirm = events[(events.side == "LOW") & (events.event_type == "STRUCTURE_CONFIRM")]
    actionable = pd.concat([high_watch, low_confirm], ignore_index=True)
    base["trend_distribution_bars"] = {
        str(k): int(v) for k, v in x.trend_state.value_counts().to_dict().items()
    }
    base["ACTIONABLE_BY_TREND"] = metrics_by_trend(actionable)
    return base


def main():
    x = build_scored_frame()
    events = build_events(x)
    events.to_csv(RESULTS / "events_v15.csv", index=False)
    summary = summarize(x, events)
    (RESULTS / "summary_events_v15.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V1.5 EVENT SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
