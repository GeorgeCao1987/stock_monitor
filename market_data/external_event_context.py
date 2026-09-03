from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

# External realtime-event context is intentionally separated from market-price priors.
# This module does NOT change V1.8/V1.9 frozen thresholds by itself.  It only
# standardizes the information available at decision time so that an incremental
# baseline-vs-news backtest can decide whether the layer deserves model weight.

EVENT_COLUMNS = [
    "event_direction",       # -1 bearish, 0 neutral/mixed, +1 bullish
    "event_strength",        # 0..1 estimated impact magnitude
    "event_scope",           # GLOBAL / ASIA / CHINA / AI_SEMI / PCB / SINGLE_NAME / OTHER
    "event_freshness",       # 0..1, decays as event gets older
    "event_confidence",      # 0..1 source/evidence confidence
    "affected_chain",        # semicolon-separated chain tags
    "event_source_count",    # corroborating source count
    "event_active",          # whether event was public and still relevant at snapshot time
]

DEFAULTS = {
    "event_direction": 0,
    "event_strength": 0.0,
    "event_scope": "NONE",
    "event_freshness": 0.0,
    "event_confidence": 0.0,
    "affected_chain": "",
    "event_source_count": 0,
    "event_active": False,
}


def _clip01(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def normalize_event_context(ctx: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize event-context rows without using any future information.

    Expected time keys are either `day` for a 10:00 daily snapshot or `ts` for
    intraday event-time snapshots.  Callers must only provide events that were
    already public at that row's decision timestamp.
    """
    if ctx is None or len(ctx) == 0:
        return pd.DataFrame(columns=["day", *EVENT_COLUMNS, "event_effective_score", "event_bucket"])

    z = ctx.copy()
    if "day" not in z.columns and "ts" in z.columns:
        z["day"] = pd.to_datetime(z["ts"]).dt.date
    elif "day" in z.columns:
        z["day"] = pd.to_datetime(z["day"]).dt.date
    else:
        raise ValueError("event context requires `day` or `ts`")

    for col, default in DEFAULTS.items():
        if col not in z.columns:
            z[col] = default

    z["event_direction"] = pd.to_numeric(z["event_direction"], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    z["event_strength"] = _clip01(z["event_strength"])
    z["event_freshness"] = _clip01(z["event_freshness"])
    z["event_confidence"] = _clip01(z["event_confidence"])
    z["event_source_count"] = pd.to_numeric(z["event_source_count"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    z["event_active"] = z["event_active"].fillna(False).astype(bool)
    z["event_scope"] = z["event_scope"].fillna("NONE").astype(str)
    z["affected_chain"] = z["affected_chain"].fillna("").astype(str)

    active = z["event_active"].astype(float)
    z["event_effective_score"] = (
        z["event_direction"].astype(float)
        * z["event_strength"]
        * z["event_freshness"]
        * z["event_confidence"]
        * active
    )
    z["event_bucket"] = np.select(
        [z["event_effective_score"] >= 0.20, z["event_effective_score"] <= -0.20],
        ["POSITIVE", "NEGATIVE"],
        default="NEUTRAL",
    )
    return z


def attach_daily_event_context(daily: pd.DataFrame, ctx: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Attach the latest/public event snapshot to daily/opening rows.

    If no historical event dataset is supplied, rows stay explicit NEUTRAL
    rather than silently pretending the news layer was tested.
    """
    z = daily.copy()
    if "day" not in z.columns:
        raise ValueError("daily frame requires `day`")
    z["day"] = pd.to_datetime(z["day"]).dt.date

    n = normalize_event_context(ctx)
    if n.empty:
        for col, default in DEFAULTS.items():
            z[col] = default
        z["event_effective_score"] = 0.0
        z["event_bucket"] = "NEUTRAL"
        z["event_context_available"] = False
        return z

    # Multiple headlines can exist in one snapshot.  Keep the strongest absolute
    # effective event for now; aggregation policy itself must be validated later.
    n = n.assign(_abs=n["event_effective_score"].abs()).sort_values(["day", "_abs"])
    n = n.groupby("day", as_index=False).tail(1).drop(columns="_abs")
    keep = ["day", *EVENT_COLUMNS, "event_effective_score", "event_bucket"]
    z = z.merge(n[keep], on="day", how="left")

    for col, default in DEFAULTS.items():
        z[col] = z[col].fillna(default)
    z["event_effective_score"] = z["event_effective_score"].fillna(0.0)
    z["event_bucket"] = z["event_bucket"].fillna("NEUTRAL")
    z["event_context_available"] = z["event_active"].astype(bool)
    return z


def event_context_schema() -> dict:
    return {
        "separate_from_price_prior": True,
        "future_leakage_forbidden": True,
        "decision_time_rule": "Only events already public by the snapshot/event timestamp may be attached.",
        "fields": {k: DEFAULTS[k] for k in EVENT_COLUMNS},
        "effective_score_formula": "direction * strength * freshness * confidence * active",
        "current_model_weight": 0.0,
        "promotion_rule": "Add weight only after baseline-vs-event incremental backtest is stable across periods and holdout.",
    }
