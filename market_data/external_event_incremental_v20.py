from pathlib import Path
import json
import pandas as pd

import event_engine_v14 as e14
from opening_regime_diagnostics import add_opening_regime
from opening_extreme_forecast_v18 import add_extreme_forecast
from external_event_context import load_raw_event_file, event_context_schema

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

EVENT_THRESHOLD = 0.20


def prediction_metrics(z: pd.DataFrame, pred_col: str) -> dict:
    total = len(z)
    out = {"days": int(total)}
    for state, label in [("HIGH_AHEAD", "high_after_10"), ("LOW_AHEAD", "low_after_10")]:
        q = z[z[pred_col] == state]
        out[state] = {
            "signals": int(len(q)),
            "coverage": float(len(q) / total) if total else None,
            "precision": float(q[label].mean()) if len(q) else None,
        }
    out["uncertain"] = int((z[pred_col] == "UNCERTAIN").sum())
    return out


def apply_uncertain_fill(z: pd.DataFrame) -> pd.Series:
    p = z["forecast"].copy()
    p = p.mask((p == "UNCERTAIN") & (z.event_effective_score >= EVENT_THRESHOLD), "HIGH_AHEAD")
    p = p.mask((p == "UNCERTAIN") & (z.event_effective_score <= -EVENT_THRESHOLD), "LOW_AHEAD")
    return p


def apply_conflict_guard(z: pd.DataFrame) -> pd.Series:
    # Diagnostic abstention only: if strong realtime news conflicts with the
    # frozen price prior, mark it uncertain instead of reversing direction.
    p = z["forecast"].copy()
    p = p.mask((p == "HIGH_AHEAD") & (z.event_effective_score <= -EVENT_THRESHOLD), "UNCERTAIN")
    p = p.mask((p == "LOW_AHEAD") & (z.event_effective_score >= EVENT_THRESHOLD), "UNCERTAIN")
    return p


def main():
    raw_events = load_raw_event_file()
    x = e14.build_scored_frame()
    _, daily0 = add_opening_regime(x)

    baseline = add_extreme_forecast(daily0, event_context=None)
    z = add_extreme_forecast(daily0, event_context=raw_events)
    z["pred_baseline"] = baseline["forecast"].values
    z["pred_uncertain_fill"] = apply_uncertain_fill(z)
    z["pred_conflict_guard"] = apply_conflict_guard(z)

    stratified = {}
    for forecast in ["HIGH_AHEAD", "LOW_AHEAD", "UNCERTAIN"]:
        stratified[forecast] = {}
        for bucket in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            q = z[(z.forecast == forecast) & (z.event_bucket == bucket)]
            stratified[forecast][bucket] = {
                "days": int(len(q)),
                "high_after_10_rate": float(q.high_after_10.mean()) if len(q) else None,
                "low_after_10_rate": float(q.low_after_10.mean()) if len(q) else None,
            }

    report = {
        "version": "external-event-incremental-v20",
        "status": "development_diagnostic_not_frozen",
        "event_context": event_context_schema(),
        "event_threshold_diagnostic": EVENT_THRESHOLD,
        "raw_external_event_rows": int(len(raw_events)) if raw_events is not None else 0,
        "event_context_available_days": int(z.event_context_available.sum()),
        "event_context_available_rate": float(z.event_context_available.mean()) if len(z) else None,
        "baseline": prediction_metrics(z, "pred_baseline"),
        "candidate_uncertain_fill": prediction_metrics(z, "pred_uncertain_fill"),
        "candidate_conflict_guard": prediction_metrics(z, "pred_conflict_guard"),
        "stratified_information": stratified,
        "rules": {
            "uncertain_fill": "Only fill baseline UNCERTAIN when event score >= +0.20 or <= -0.20.",
            "conflict_guard": "Do not reverse baseline; abstain to UNCERTAIN when a strong event conflicts.",
            "promotion": "Neither candidate may enter production until cross-period stability and a new untouched holdout are passed.",
        },
        "holdout_warning": "2025-11..12 has already been seen and cannot validate the newly designed event layer.",
    }

    z.to_csv(RESULTS / "external_event_incremental_v20.csv", index=False)
    (RESULTS / "external_event_incremental_v20.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTERNAL EVENT INCREMENTAL V2.0")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
