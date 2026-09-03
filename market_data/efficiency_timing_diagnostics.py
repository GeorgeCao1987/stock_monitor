from pathlib import Path
import json
import numpy as np
import pandas as pd

import backtest_v13 as v13
import backtest_v14 as v14
import event_engine_v14 as e14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"


def add_efficiency(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy().sort_values("ts").reset_index(drop=True)
    day = z.ts.dt.date
    z["ret1_day"] = z.groupby(day).close.pct_change()
    amount = pd.to_numeric(z.amount, errors="coerce").replace(0, np.nan)
    z["impact_eff"] = z.ret1_day.abs() / amount
    z["prior_eff3"] = z.groupby(day).impact_eff.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    z["eff_ratio"] = z.impact_eff / z.prior_eff3.replace(0, np.nan)
    z["prior_amount3"] = z.groupby(day).amount.transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=2).mean()
    )
    z["amount_vs_prior3"] = amount / z.prior_amount3.replace(0, np.nan)
    # This is descriptive only; no threshold is used to create/filter a signal.
    z["eff_decay_with_more_money"] = (z.eff_ratio < 1.0) & (z.amount_vs_prior3 > 1.0)
    return z


def build():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = v14.add_v14_features(target)
    x = v14.add_context(x)
    x = v14.score_states(x)
    x = add_efficiency(x)
    x["prev_high"] = x.groupby(x.ts.dt.date).high.shift(1)
    x["prev_low"] = x.groupby(x.ts.dt.date).low.shift(1)
    events = e14.build_events(x)
    ctx = x[["ts", "eff_ratio", "amount_vs_prior3", "eff_decay_with_more_money", "atr_pct"]]
    events = events.merge(ctx, on="ts", how="left", suffixes=("", "_ctx"))
    return x, events


def basic(z: pd.DataFrame):
    if z.empty:
        return {"n": 0}
    valid = z.atr_pct.notna() & np.isfinite(z.atr_pct) & (z.atr_pct > 0)
    a = z[valid]
    denom = a.mae_30m.where(a.mae_30m > 1e-9, np.nan)
    return {
        "n": int(len(z)),
        "fixed_1_5": float((z.future_30m >= .015).mean()),
        "atr_0_75": float((a.mfe_30m >= .75 * a.atr_pct).mean()) if len(a) else None,
        "directional": float((z.mfe_30m > z.mae_30m).mean()),
        "median_mfe": float(z.mfe_30m.median()),
        "median_mae": float(z.mae_30m.median()),
        "median_ratio": float((a.mfe_30m / denom).median()) if len(a) else None,
        "median_eff_ratio": float(z.eff_ratio.median()) if z.eff_ratio.notna().any() else None,
        "median_amount_vs_prior3": float(z.amount_vs_prior3.median()) if z.amount_vs_prior3.notna().any() else None,
    }


def efficiency_groups(events: pd.DataFrame, side="HIGH", etype="WATCH_START"):
    z = events[(events.side == side) & (events.event_type == etype)].copy()
    out = {"ALL": basic(z)}
    e = z[z.eff_ratio.notna()].copy()
    if len(e) >= 9:
        # Fixed semantic bands around 1.0, not data-mined quantiles.
        out["EFF_LT_0_5"] = basic(e[e.eff_ratio < .5])
        out["EFF_0_5_TO_1"] = basic(e[(e.eff_ratio >= .5) & (e.eff_ratio < 1.0)])
        out["EFF_GE_1"] = basic(e[e.eff_ratio >= 1.0])
    out["DECAY_MORE_MONEY"] = basic(z[z.eff_decay_with_more_money.fillna(False)])
    out["NO_DECAY_MORE_MONEY"] = basic(z[~z.eff_decay_with_more_money.fillna(False)])
    return out


def high_event_pairs(events: pd.DataFrame):
    h = events[events.side == "HIGH"].copy().sort_values(["event_id", "ts"])
    rows = []
    for eid, g in h.groupby("event_id"):
        w = g[g.event_type == "WATCH_START"]
        if w.empty:
            continue
        wr = w.iloc[0]
        c = g[(g.event_type == "STRUCTURE_CONFIRM") & (g.ts >= wr.ts)]
        if c.empty:
            delay = None
            cr = None
        else:
            cr = c.iloc[0]
            delay = int(round((cr.ts - wr.ts).total_seconds() / 300.0))
        rows.append({
            "event_id": eid,
            "watch_ts": wr.ts,
            "confirm_ts": None if cr is None else cr.ts,
            "confirm_delay_bars": delay,
            "watch_future_30m": wr.future_30m,
            "watch_mfe_30m": wr.mfe_30m,
            "watch_mae_30m": wr.mae_30m,
            "watch_atr_pct": wr.atr_pct,
            "watch_eff_ratio": wr.eff_ratio,
            "watch_amount_vs_prior3": wr.amount_vs_prior3,
            "confirm_future_30m": None if cr is None else cr.future_30m,
            "confirm_mfe_30m": None if cr is None else cr.mfe_30m,
            "confirm_mae_30m": None if cr is None else cr.mae_30m,
            "confirm_atr_pct": None if cr is None else cr.atr_pct,
        })
    return pd.DataFrame(rows)


def pair_metrics(pairs: pd.DataFrame):
    if pairs.empty:
        return {}
    out = {
        "events": int(len(pairs)),
        "confirmed_any": int(pairs.confirm_delay_bars.notna().sum()),
        "confirm_rate": float(pairs.confirm_delay_bars.notna().mean()),
    }
    for label, mask in [
        ("NO_CONFIRM", pairs.confirm_delay_bars.isna()),
        ("CONFIRM_1_BAR", pairs.confirm_delay_bars == 1),
        ("CONFIRM_2_BARS", pairs.confirm_delay_bars == 2),
        ("CONFIRM_1_2_BARS", pairs.confirm_delay_bars.isin([1,2])),
        ("CONFIRM_3PLUS", pairs.confirm_delay_bars >= 3),
    ]:
        z = pairs[mask].copy()
        if z.empty:
            out[label] = {"n": 0}
            continue
        valid = z.watch_atr_pct.notna() & (z.watch_atr_pct > 0)
        a = z[valid]
        out[label] = {
            "n": int(len(z)),
            "watch_fixed_1_5": float((z.watch_future_30m >= .015).mean()),
            "watch_atr_0_75": float((a.watch_mfe_30m >= .75 * a.watch_atr_pct).mean()) if len(a) else None,
            "watch_directional": float((z.watch_mfe_30m > z.watch_mae_30m).mean()),
            "watch_median_ratio": float((a.watch_mfe_30m / a.watch_mae_30m.replace(0,np.nan)).median()) if len(a) else None,
        }
    return out


def main():
    x, events = build()
    pairs = high_event_pairs(events)
    report = {
        "trading_days": int(x.ts.dt.date.nunique()),
        "HIGH_WATCH_EFFICIENCY": efficiency_groups(events, "HIGH", "WATCH_START"),
        "LOW_CONFIRM_EFFICIENCY": efficiency_groups(events, "LOW", "STRUCTURE_CONFIRM"),
        "HIGH_CONFIRM_TIMING": pair_metrics(pairs),
        "note": "All efficiency and timing features use only information available at or before each event; future data is used only for scoring.",
    }
    events.to_csv(RESULTS / "efficiency_timing_events.csv", index=False)
    pairs.to_csv(RESULTS / "high_event_pairs.csv", index=False)
    (RESULTS / "efficiency_timing_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EFFICIENCY TIMING DIAGNOSTICS")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
