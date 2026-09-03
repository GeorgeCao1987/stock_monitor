from pathlib import Path
import json
import numpy as np
import pandas as pd

import backtest_v13 as v13

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"
WATCH_THRESHOLD = 2.0
CONFIRM_THRESHOLD = 2.5


def add_v14_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("ts").reset_index(drop=True)
    day = x.ts.dt.date
    x["bar_idx"] = x.groupby(day).cumcount()
    x["day_open"] = x.groupby(day).open.transform("first")
    x["ret_from_open"] = x.close / x.day_open - 1
    x["typical"] = (x.high + x.low + x.close) / 3
    x["cum_pv"] = (x.typical * x.volume).groupby(day).cumsum()
    x["cum_v"] = x.volume.groupby(day).cumsum().replace(0, np.nan)
    x["vwap"] = x.cum_pv / x.cum_v
    x["dist_vwap"] = (x.close - x.vwap) / x.vwap

    x["prior_cum_high"] = x.groupby(day).high.transform(lambda s: s.shift(1).cummax())
    x["prior_cum_low"] = x.groupby(day).low.transform(lambda s: s.shift(1).cummin())
    x["cum_high"] = x.groupby(day).high.cummax()
    x["cum_low"] = x.groupby(day).low.cummin()
    x["pos_in_range"] = (x.close - x.cum_low) / (x.cum_high - x.cum_low).replace(0, np.nan)

    prev_close = x.close.shift(1)
    tr = pd.concat([(x.high - x.low), (x.high - prev_close).abs(), (x.low - prev_close).abs()], axis=1).max(axis=1)
    x["atr12"] = x.groupby(day).apply(
        lambda g: tr.loc[g.index].shift(1).rolling(12, min_periods=4).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    x["atr_pct"] = x.atr12 / x.close

    x["vol_ma6_prior"] = x.groupby(day).volume.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    x["vol_ratio"] = x.volume / x.vol_ma6_prior
    x["amount_ma6_prior"] = x.groupby(day).amount.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    x["amount_ratio"] = x.amount / x.amount_ma6_prior

    rng = (x.high - x.low).replace(0, np.nan)
    x["upper_wick_ratio"] = (x.high - x[["open", "close"]].max(axis=1)) / rng
    x["lower_wick_ratio"] = (x[["open", "close"]].min(axis=1) - x.low) / rng
    x["bar_body_abs"] = (x.close - x.open).abs() / x.close
    x["body_shrink"] = x.bar_body_abs / x.groupby(day).bar_body_abs.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )

    x["ret1"] = x.groupby(day).close.pct_change()
    x["ret3"] = x.groupby(day).close.pct_change(3)
    x["ret6"] = x.groupby(day).close.pct_change(6)
    x["ret3_accel"] = x.groupby(day).ret3.diff()
    x["new_high"] = x.high >= x.prior_cum_high.fillna(-np.inf)
    x["new_low"] = x.low <= x.prior_cum_low.fillna(np.inf)
    x["dist_to_high_atr"] = (x.prior_cum_high - x.close) / x.atr12
    x["dist_to_low_atr"] = (x.close - x.prior_cum_low) / x.atr12
    return x


def add_context(x: pd.DataFrame) -> pd.DataFrame:
    pcb = v13.build_pcb_context()
    idx = v13.build_index_context()
    x = x.merge(pcb, on="ts", how="left").merge(idx, on="ts", how="left")
    x["pcb_rel"] = x.pcb_ret - x.index_ret
    x["pcb_rel_chg3"] = x.pcb_rel.diff(3)
    x["pcb_breadth_chg3"] = x.pcb_up_breadth.diff(3)
    for frag, prefix in [
        ("000660_KS", "hynix"), ("005930_KS", "samsung"), ("IDX_KS11", "kospi"),
        ("NQ_F", "nq"), ("CL_F", "oil")
    ]:
        x = v13.merge_external(x, frag, prefix)
    x = v13.attach_macro_history(x)
    x = v13.attach_news(x)
    return x


def score_states(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    high_loc = (z.pos_in_range >= .72) | (z.dist_to_high_atr <= .35) | z.new_high
    low_loc = (z.pos_in_range <= .28) | (z.dist_to_low_atr <= .35) | z.new_low
    high_ext = (z.ret6 >= .006) | (z.ret_from_open >= .010) | (z.ret3 >= .005)
    low_ext = (z.ret6 <= -.006) | (z.ret_from_open <= -.010) | (z.ret3 <= -.005)

    hs = high_loc.astype(float) * .70 + high_ext.astype(float) * .60
    ls = low_loc.astype(float) * .70 + low_ext.astype(float) * .60
    hs += z.new_high.astype(float) * .45
    ls += z.new_low.astype(float) * .45
    hs += (z.upper_wick_ratio >= .35).astype(float) * .35
    ls += (z.lower_wick_ratio >= .35).astype(float) * .35
    hs += (z.vol_ratio <= .90).astype(float) * .40
    ls += (z.vol_ratio <= .90).astype(float) * .40
    hs += (z.body_shrink <= .75).astype(float) * .25
    ls += (z.body_shrink <= .75).astype(float) * .25
    hs += (z.ret3_accel < 0).astype(float) * .25
    ls += (z.ret3_accel > 0).astype(float) * .25
    hs += (z.pcb_rel_chg3 < -.0015).astype(float) * .50
    ls += (z.pcb_rel_chg3 > .0015).astype(float) * .50
    hs += (z.pcb_breadth_chg3 < 0).astype(float) * .25
    ls += (z.pcb_breadth_chg3 > 0).astype(float) * .25
    hs += (z.dist_vwap > .006).astype(float) * .30
    ls += (z.dist_vwap < -.006).astype(float) * .30
    hs += (z.atr_pct > .008).astype(float) * .20
    ls += (z.atr_pct > .008).astype(float) * .20

    korea_down = sum((z[c] < 0).fillna(False).astype(int) for c in ["hynix_ret", "samsung_ret", "kospi_ret"])
    korea_up = sum((z[c] > 0).fillna(False).astype(int) for c in ["hynix_ret", "samsung_ret", "kospi_ret"])
    hs += korea_down * .05 + ((z.nq_ret < 0).fillna(False)).astype(float) * .10
    ls += korea_up * .05 + ((z.nq_ret > 0).fillna(False)).astype(float) * .10

    oil_up = ((z.oil_ret > 0).fillna(False)) | ((z.oil_mom3d > 0).fillna(False))
    stress = oil_up & (z.tyx_pct_3y >= .80).fillna(False)
    hs += (stress & ((z.pcb_rel_chg3 < 0) | (korea_down >= 2))).astype(float) * .10
    ls -= stress.astype(float) * .10

    hs -= ((z.ret3 > .012) & (z.vol_ratio > 1.25) & (z.pcb_rel_chg3 > 0)).astype(float) * .50
    ls -= ((z.ret3 < -.012) & (z.vol_ratio > 1.25) & (z.pcb_rel_chg3 < 0)).astype(float) * .50

    z["high_score"] = hs
    z["low_score"] = ls
    eligible = z.bar_idx >= 3
    z["high_watch"] = eligible & high_loc & high_ext & (hs >= WATCH_THRESHOLD)
    z["low_watch"] = eligible & low_loc & low_ext & (ls >= WATCH_THRESHOLD)
    z["high_confirm"] = eligible & high_loc & high_ext & (hs >= CONFIRM_THRESHOLD)
    z["low_confirm"] = eligible & low_loc & low_ext & (ls >= CONFIRM_THRESHOLD)
    return z


def future_metrics(x, i, side):
    return v13.evaluate_future(x, i, side)


def _eligible_extreme_metrics(x, side, signal_col):
    out = {
        "opening_or_unforecastable_extremes": 0,
        "eligible_extremes": 0,
        "pre_1bar_recall": None,
        "pre_2bar_recall": None,
        "pre_3bar_recall": None,
        "median_earliest_lead_bars_within_3": None,
        "median_closest_lead_bars_within_3": None,
    }
    hits = {1: 0, 2: 0, 3: 0}
    den = {1: 0, 2: 0, 3: 0}
    earliest, closest = [], []
    for _, g0 in x.groupby(x.ts.dt.date):
        g = g0.sort_values("ts").reset_index(drop=True)
        pos = int(g.high.idxmax()) if side == "HIGH" else int(g.low.idxmin())
        if pos < 3:
            out["opening_or_unforecastable_extremes"] += 1
        else:
            out["eligible_extremes"] += 1
        for bars in [1, 2, 3]:
            if pos < bars:
                continue
            den[bars] += 1
            w = g.iloc[pos-bars:pos]
            if bool(w[signal_col].any()):
                hits[bars] += 1
        if pos >= 3:
            w = g.iloc[pos-3:pos].reset_index(drop=True)
            p = np.where(w[signal_col].to_numpy(dtype=bool))[0]
            if len(p):
                leads = 3 - p
                earliest.append(int(leads.max()))
                closest.append(int(leads.min()))
    for bars in [1, 2, 3]:
        out[f"pre_{bars}bar_recall"] = hits[bars] / den[bars] if den[bars] else None
    if earliest:
        out["median_earliest_lead_bars_within_3"] = float(np.median(earliest))
        out["median_closest_lead_bars_within_3"] = float(np.median(closest))
    return out


def build_signal_rows(x):
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
                "ts": r.ts, "side": side, "tier": tier, "price": r.close,
                "score": r[score_col], "ret_from_open": r.ret_from_open,
                "dist_vwap": r.dist_vwap, "pos_in_range": r.pos_in_range,
                "atr_pct": r.atr_pct, "vol_ratio": r.vol_ratio,
                "amount_ratio": r.amount_ratio, "upper_wick_ratio": r.upper_wick_ratio,
                "lower_wick_ratio": r.lower_wick_ratio, "ret3": r.ret3, "ret6": r.ret6,
                "pcb_rel": r.pcb_rel, "pcb_rel_chg3": r.pcb_rel_chg3,
                "pcb_breadth": r.pcb_up_breadth, "pcb_breadth_chg3": r.pcb_breadth_chg3,
                "nq_ret": r.nq_ret, "oil_ret": r.oil_ret,
                "tyx_pct": r.tyx_pct_3y, "news_score_60m": r.news_score_60m,
            }
            rec.update(future_metrics(x, i, side))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(x, signals):
    summary = {"thresholds": {"watch": WATCH_THRESHOLD, "confirm": CONFIRM_THRESHOLD}}
    for side in ["HIGH", "LOW"]:
        summary[side] = {}
        for tier in ["WATCH", "CONFIRM"]:
            a = signals[(signals.side == side) & (signals.tier == tier)]
            key = tier.lower()
            summary[side][key] = {
                "signal_bars": int(len(a)),
                "win_15m_1pct": float((a.future_15m >= .01).mean()) if len(a) else None,
                "win_30m_1_5pct": float((a.future_30m >= .015).mean()) if len(a) else None,
                "median_mfe_30m": float(a.mfe_30m.median()) if len(a) else None,
                "median_mae_30m": float(a.mae_30m.median()) if len(a) else None,
            }
            col = f"{side.lower()}_{key}"
            summary[side][key].update(_eligible_extreme_metrics(x, side, col))
    return summary


def main():
    target = v13.load_a(TARGET)
    if target.empty:
        raise SystemExit("missing target data")
    x = add_v14_features(target)
    x = add_context(x)
    x = score_states(x)
    signals = build_signal_rows(x)
    signals.to_csv(RESULTS / "signals_v14.csv", index=False)
    summary = summarize(x, signals)
    (RESULTS / "summary_v14.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("V1.4 SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
