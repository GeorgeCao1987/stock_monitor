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


def build_frame():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = v14.add_v14_features(target)
    x = v14.add_context(x)
    # Important: V1.6 keeps V1.4 mechanical scores/signals unchanged.
    x = v14.score_states(x)
    # Trend is annotation/action context only; it never removes a candidate.
    x = v15.add_trend_regime(x)
    x = x.sort_values("ts").reset_index(drop=True)
    x["prev_high"] = x.groupby(x.ts.dt.date).high.shift(1)
    x["prev_low"] = x.groupby(x.ts.dt.date).low.shift(1)
    return x


def classify_action(row):
    side = row["side"]
    etype = row["event_type"]
    trend = row["trend_state"]

    # Frozen from Mar-Aug pooled diagnostics before Jan-Feb holdout:
    # 1) RANGE high WATCH was the only high-side regime with strong, repeated edge.
    # 2) DOWN low STRUCTURE_CONFIRM was the most stable sufficiently-sized low-side group.
    if side == "HIGH" and etype == "WATCH_START" and trend == "RANGE":
        return "EXECUTE_REDUCE_OR_REVERSE_T", "A"
    if side == "LOW" and etype == "STRUCTURE_CONFIRM" and trend == "DOWN":
        return "EXECUTE_LOW_ENTRY_T", "B"

    if side == "HIGH" and etype == "WATCH_START" and trend == "UP":
        return "OBSERVE_TOP_RISK_TREND_CONTINUATION", "C"
    if side == "LOW" and etype == "WATCH_START" and trend == "UP":
        return "OBSERVE_UPTREND_PULLBACK", "C"
    return "OBSERVE_ONLY", "C"


def build_events(x):
    events = e14.build_events(x)
    if events.empty:
        return events
    ctx = x[["ts", "trend_state", "up_votes", "down_votes", "atr_pct"]].copy()
    events = events.merge(ctx, on="ts", how="left", suffixes=("", "_ctx"))
    labels = events.apply(classify_action, axis=1, result_type="expand")
    events["policy_action"] = labels[0]
    events["policy_grade"] = labels[1]
    events["policy_actionable"] = events.policy_grade.isin(["A", "B"])
    return events


def metrics(z):
    return e14.metrics(z)


def summarize(x, events):
    high_all = events[(events.side == "HIGH") & (events.event_type == "WATCH_START")]
    low_all = events[(events.side == "LOW") & (events.event_type == "STRUCTURE_CONFIRM")]
    baseline_actionable = pd.concat([high_all, low_all], ignore_index=True)
    policy = events[events.policy_actionable]
    grade_a = events[events.policy_grade == "A"]
    grade_b = events[events.policy_grade == "B"]

    return {
        "model": "V1.6",
        "architecture": "V1.4 candidates/events unchanged; UP/DOWN/RANGE only changes action grade",
        "frozen_policy": {
            "A": "HIGH WATCH_START in RANGE -> execute reduce/reverse T",
            "B": "LOW STRUCTURE_CONFIRM in DOWN -> execute low-entry T",
            "C": "all other events remain observable, not mechanically executed",
        },
        "trading_days": int(x.ts.dt.date.nunique()),
        "trend_bars": {str(k): int(v) for k, v in x.trend_state.value_counts().to_dict().items()},
        "BASELINE_V14_ACTIONABLE": metrics(baseline_actionable),
        "V16_ACTIONABLE": metrics(policy),
        "GRADE_A": metrics(grade_a),
        "GRADE_B": metrics(grade_b),
        "counts": {
            "all_events": int(len(events)),
            "baseline_actionable": int(len(baseline_actionable)),
            "v16_actionable": int(len(policy)),
            "grade_a": int(len(grade_a)),
            "grade_b": int(len(grade_b)),
        },
        "candidate_generation_note": "No V1.4 candidate/watch/confirm flag is filtered by trend; daily candidate recall is preserved.",
    }


def main():
    x = build_frame()
    events = build_events(x)
    events.to_csv(RESULTS / "events_v16.csv", index=False)
    summary = summarize(x, events)
    (RESULTS / "summary_events_v16.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V1.6 EVENT SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
