from __future__ import annotations

from pathlib import Path
from typing import Optional
import math
import numpy as np
import pandas as pd

# External realtime-event context is intentionally separated from market-price priors.
# This module does NOT change V1.8/V1.9 frozen thresholds by itself. It only
# standardizes information that was already public at each decision timestamp,
# so an incremental baseline-vs-news backtest can decide whether the layer earns weight.

SH_TZ = "Asia/Shanghai"
DEFAULT_LOOKBACK_HOURS = 18.0
DEFAULT_HALF_LIFE_HOURS = 6.0

EVENT_COLUMNS = [
    "event_direction",       # -1 bearish, 0 neutral/mixed, +1 bullish
    "event_strength",        # 0..1 estimated impact magnitude
    "event_scope",           # GLOBAL / ASIA / CHINA / AI_SEMI / PCB / SINGLE_NAME / OTHER
    "event_freshness",       # 0..1, recomputed at the decision timestamp when published_ts exists
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
    """Normalize already time-valid event-context rows.

    Time filtering is done by snapshot builders below when raw `published_ts`
    headlines are supplied. If callers supply pre-aggregated context, they are
    responsible for ensuring it contains no information published after the
    relevant decision time.
    """
    if ctx is None or len(ctx) == 0:
        return pd.DataFrame(columns=[*EVENT_COLUMNS, "event_effective_score", "event_bucket"])

    z = ctx.copy()
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


def _to_utc_ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize(SH_TZ)
    return t.tz_convert("UTC")


def _strongest_public_event(
    raw_events: pd.DataFrame,
    snapshot_ts,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
) -> dict:
    """Return strongest event that was public by snapshot_ts.

    Future headlines are mechanically excluded. Freshness decays exponentially
    from publication time and is diagnostic-only until validated.
    """
    if raw_events is None or raw_events.empty or "published_ts" not in raw_events.columns:
        return {**DEFAULTS, "event_effective_score": 0.0, "event_bucket": "NEUTRAL", "event_context_available": False}

    snap = _to_utc_ts(snapshot_ts)
    z = raw_events.copy()
    z["_published"] = pd.to_datetime(z["published_ts"], utc=True, errors="coerce")
    z = z[z["_published"].notna() & (z["_published"] <= snap)].copy()
    if z.empty:
        return {**DEFAULTS, "event_effective_score": 0.0, "event_bucket": "NEUTRAL", "event_context_available": False}

    age_hours = (snap - z["_published"]).dt.total_seconds() / 3600.0
    z = z[(age_hours >= 0) & (age_hours <= lookback_hours)].copy()
    if z.empty:
        return {**DEFAULTS, "event_effective_score": 0.0, "event_bucket": "NEUTRAL", "event_context_available": False}

    age_hours = (snap - z["_published"]).dt.total_seconds() / 3600.0
    z["event_freshness"] = age_hours.map(lambda h: math.exp(-math.log(2.0) * float(h) / half_life_hours))
    z["event_active"] = True
    n = normalize_event_context(z)
    n = n.assign(_abs=n["event_effective_score"].abs()).sort_values("_abs")
    row = n.iloc[-1]
    out = {col: row[col] for col in EVENT_COLUMNS}
    out["event_effective_score"] = float(row["event_effective_score"])
    out["event_bucket"] = str(row["event_bucket"])
    out["event_context_available"] = True
    if "title" in z.columns:
        idx = row.name
        if idx in z.index:
            out["event_title"] = str(z.loc[idx, "title"] or "")
            out["event_published_ts"] = str(z.loc[idx, "published_ts"] or "")
    return out


def attach_daily_event_context(
    daily: pd.DataFrame,
    ctx: Optional[pd.DataFrame] = None,
    snapshot_time: str = "10:00",
) -> pd.DataFrame:
    """Attach an opening-time external-message snapshot to daily rows.

    Raw headlines with `published_ts` are filtered at each trading day's 10:00
    decision time. Pre-aggregated day-level contexts are also supported.
    """
    z = daily.copy()
    if "day" not in z.columns:
        raise ValueError("daily frame requires `day`")
    z["day"] = pd.to_datetime(z["day"]).dt.date

    if ctx is None or len(ctx) == 0:
        for col, default in DEFAULTS.items():
            z[col] = default
        z["event_effective_score"] = 0.0
        z["event_bucket"] = "NEUTRAL"
        z["event_context_available"] = False
        return z

    if "published_ts" in ctx.columns:
        rows = []
        for day in z["day"]:
            snap = pd.Timestamp(f"{day} {snapshot_time}", tz=SH_TZ)
            rows.append(_strongest_public_event(ctx, snap))
        c = pd.DataFrame(rows, index=z.index)
        for col in c.columns:
            z[col] = c[col]
        return z

    n = ctx.copy()
    if "day" not in n.columns and "ts" in n.columns:
        n["day"] = pd.to_datetime(n["ts"]).dt.date
    elif "day" in n.columns:
        n["day"] = pd.to_datetime(n["day"]).dt.date
    else:
        raise ValueError("pre-aggregated event context requires `day`, `ts`, or raw `published_ts`")
    n = normalize_event_context(n)
    n["day"] = pd.to_datetime(ctx["day"] if "day" in ctx.columns else ctx["ts"]).dt.date
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


def attach_intraday_event_context(
    rows: pd.DataFrame,
    ctx: Optional[pd.DataFrame] = None,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Attach the strongest already-public external event at each intraday row."""
    z = rows.copy()
    if ts_col not in z.columns:
        raise ValueError(f"intraday frame requires `{ts_col}`")
    if ctx is None or len(ctx) == 0:
        for col, default in DEFAULTS.items():
            z[col] = default
        z["event_effective_score"] = 0.0
        z["event_bucket"] = "NEUTRAL"
        z["event_context_available"] = False
        return z

    if "published_ts" not in ctx.columns:
        # Fall back to day-level attachment for pre-aggregated context.
        tmp = z.copy()
        tmp["day"] = pd.to_datetime(tmp[ts_col]).dt.date
        return attach_daily_event_context(tmp, ctx)

    snapshots = [_strongest_public_event(ctx, ts) for ts in z[ts_col]]
    c = pd.DataFrame(snapshots, index=z.index)
    for col in c.columns:
        z[col] = c[col]
    return z


def load_raw_event_file(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    p = path or (Path(__file__).resolve().parent / "data" / "news_events.csv")
    if not p.exists():
        return None
    x = pd.read_csv(p)
    return x if len(x) else None


def event_context_schema() -> dict:
    return {
        "separate_from_price_prior": True,
        "future_leakage_forbidden": True,
        "decision_time_rule": "Only events already public by the opening/event timestamp may be attached.",
        "opening_snapshot": "10:00 Asia/Shanghai",
        "intraday_snapshot": "event timestamp",
        "default_lookback_hours": DEFAULT_LOOKBACK_HOURS,
        "freshness_half_life_hours": DEFAULT_HALF_LIFE_HOURS,
        "fields": {k: DEFAULTS[k] for k in EVENT_COLUMNS},
        "effective_score_formula": "direction * strength * freshness * confidence * active",
        "current_model_weight": 0.0,
        "promotion_rule": "Add weight only after baseline-vs-event incremental backtest is stable across periods and untouched holdout.",
    }
