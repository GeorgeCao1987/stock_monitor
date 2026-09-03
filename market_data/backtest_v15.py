from pathlib import Path
import json
import numpy as np
import pandas as pd

import backtest_v13 as v13
import backtest_v14 as v14

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"


def _roll_by_day(series, day, window, min_periods, fn="mean"):
    s = pd.Series(series, index=series.index)
    if fn == "sum":
        return s.groupby(day).transform(lambda z: z.rolling(window, min_periods=min_periods).sum())
    return s.groupby(day).transform(lambda z: z.rolling(window, min_periods=min_periods).mean())


def add_trend_regime(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy().sort_values("ts").reset_index(drop=True)
    day = z.ts.dt.date

    z["prev_high"] = z.groupby(day).high.shift(1)
    z["prev_low"] = z.groupby(day).low.shift(1)
    z["hh"] = z.high > z.prev_high
    z["hl"] = z.low > z.prev_low
    z["lh"] = z.high < z.prev_high
    z["ll"] = z.low < z.prev_low

    hh4 = _roll_by_day(z.hh.astype(int), day, 4, 4, "sum")
    hl4 = _roll_by_day(z.hl.astype(int), day, 4, 4, "sum")
    lh4 = _roll_by_day(z.lh.astype(int), day, 4, 4, "sum")
    ll4 = _roll_by_day(z.ll.astype(int), day, 4, 4, "sum")
    z["up_structure"] = (hh4 + hl4) / 8.0
    z["down_structure"] = (lh4 + ll4) / 8.0

    above = (z.close > z.vwap).astype(float)
    z["above_vwap4"] = _roll_by_day(above, day, 4, 3, "mean")
    z["pcb_rel_ma3"] = _roll_by_day(z.pcb_rel, day, 3, 2, "mean")
    z["pcb_rel_chg3_day"] = z.groupby(day).pcb_rel.diff(3)

    price_up = z.up_structure >= .625
    price_down = z.down_structure >= .625
    vwap_up = z.above_vwap4 >= .75
    vwap_down = z.above_vwap4 <= .25
    pcb_up = z.pcb_rel_ma3 > 0
    pcb_down = z.pcb_rel_ma3 < 0

    z["up_votes"] = price_up.astype(int) + vwap_up.astype(int) + pcb_up.fillna(False).astype(int)
    z["down_votes"] = price_down.astype(int) + vwap_down.astype(int) + pcb_down.fillna(False).astype(int)

    z["trend_state"] = "RANGE"
    up_mask = (z.up_votes >= 2) & (z.up_votes > z.down_votes)
    down_mask = (z.down_votes >= 2) & (z.down_votes > z.up_votes)
    z.loc[up_mask, "trend_state"] = "UP"
    z.loc[down_mask, "trend_state"] = "DOWN"

    # These gates implement the previously defined asymmetric trend rules.
    # In an uptrend, a top warning is allowed only after both local price structure
    # and PCB relative strength have started to deteriorate.
    z["uptrend_top_break"] = (
        (z.low < z.prev_low) &
        (z.pcb_rel_chg3_day < 0)
    ).fillna(False)

    # In a downtrend, a bottom warning is allowed only after price stops making
    # a lower low, makes a higher high, and PCB relative strength improves.
    z["downtrend_bottom_turn"] = (
        (z.low >= z.prev_low) &
        (z.high > z.prev_high) &
        (z.pcb_rel_chg3_day > 0)
    ).fillna(False)
    return z


def score_states(x: pd.DataFrame) -> pd.DataFrame:
    # Freeze the V1.4 score construction; V1.5 only adds the trend-regime gate.
    z = v14.score_states(x)
    z = add_trend_regime(z)

    z["high_watch_v14"] = z.high_watch
    z["high_confirm_v14"] = z.high_confirm
    z["low_watch_v14"] = z.low_watch
    z["low_confirm_v14"] = z.low_confirm

    z["high_watch"] = z.high_watch_v14 & (
        (z.trend_state != "UP") | z.uptrend_top_break
    )
    z["high_confirm"] = z.high_confirm_v14 & (
        (z.trend_state != "UP") | z.uptrend_top_break
    )
    z["low_watch"] = z.low_watch_v14 & (
        (z.trend_state != "DOWN") | z.downtrend_bottom_turn
    )
    z["low_confirm"] = z.low_confirm_v14 & (
        (z.trend_state != "DOWN") | z.downtrend_bottom_turn
    )
    return z


def build_signal_rows(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("HIGH", "high_watch", "WATCH", "high_score"),
        ("HIGH", "high_confirm", "CONFIRM", "high_score"),
        ("LOW", "low_watch", "WATCH", "low_score"),
        ("LOW", "low_confirm", "CONFIRM", "low_score"),
    ]
    for side, flag, tier, score_col in specs:
        for i in x.index[x[flag]]:
            r = x.loc[i]
            rec = {
                "ts": r.ts,
                "side": side,
                "tier": tier,
                "price": r.close,
                "score": r[score_col],
                "trend_state": r.trend_state,
                "up_votes": r.up_votes,
                "down_votes": r.down_votes,
                "ret_from_open": r.ret_from_open,
                "dist_vwap": r.dist_vwap,
                "pos_in_range": r.pos_in_range,
                "atr_pct": r.atr_pct,
                "vol_ratio": r.vol_ratio,
                "amount_ratio": r.amount_ratio,
                "upper_wick_ratio": r.upper_wick_ratio,
                "lower_wick_ratio": r.lower_wick_ratio,
                "ret3": r.ret3,
                "ret6": r.ret6,
                "pcb_rel": r.pcb_rel,
                "pcb_rel_chg3": r.pcb_rel_chg3,
                "pcb_rel_chg3_day": r.pcb_rel_chg3_day,
                "pcb_breadth": r.pcb_up_breadth,
                "pcb_breadth_chg3": r.pcb_breadth_chg3,
            }
            rec.update(v13.evaluate_future(x, i, side))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(x, signals):
    summary = {
        "model": "V1.5",
        "change": "V1.4 scores frozen; add UP/DOWN/RANGE trend-regime gating",
        "thresholds": {"watch": v14.WATCH_THRESHOLD, "confirm": v14.CONFIRM_THRESHOLD},
        "trend_distribution": {
            str(k): int(v) for k, v in x.trend_state.value_counts().to_dict().items()
        },
    }
    for side in ["HIGH", "LOW"]:
        summary[side] = {}
        for tier in ["WATCH", "CONFIRM"]:
            a = signals[(signals.side == side) & (signals.tier == tier)] if not signals.empty else signals
            key = tier.lower()
            summary[side][key] = {
                "signal_bars": int(len(a)),
                "win_15m_1pct": float((a.future_15m >= .01).mean()) if len(a) else None,
                "win_30m_1_5pct": float((a.future_30m >= .015).mean()) if len(a) else None,
                "directional_edge_mfe_gt_mae": float((a.mfe_30m > a.mae_30m).mean()) if len(a) else None,
                "median_mfe_30m": float(a.mfe_30m.median()) if len(a) else None,
                "median_mae_30m": float(a.mae_30m.median()) if len(a) else None,
            }
            col = f"{side.lower()}_{key}"
            summary[side][key].update(v14._eligible_extreme_metrics(x, side, col))
    return summary


def main():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = v14.add_v14_features(target)
    x = v14.add_context(x)
    x = score_states(x)
    signals = build_signal_rows(x)
    signals.to_csv(RESULTS / "signals_v15.csv", index=False)
    summary = summarize(x, signals)
    (RESULTS / "summary_v15.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V1.5 SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
