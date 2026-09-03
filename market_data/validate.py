import pandas as pd
import exchange_calendars as xcals
from config import START_DATE, END_DATE

EXPECTED_BARS = 48


def expected_trade_dates():
    cal = xcals.get_calendar("XSHG")
    sessions = cal.sessions_in_range(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))
    return pd.DatetimeIndex(sessions).strftime("%Y-%m-%d")


def normalize_a_share(df):
    if df.empty:
        return df
    x = df.copy()
    x["ts"] = pd.to_datetime(x["ts"])
    if getattr(x["ts"].dt, "tz", None) is not None:
        x["ts"] = x["ts"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return x.sort_values("ts").drop_duplicates("ts", keep="last")


def validate_a_share(df, symbol):
    x = normalize_a_share(df)
    expected_dates = expected_trade_dates()
    expected_days = len(expected_dates)
    expected_total = expected_days * EXPECTED_BARS
    if x.empty:
        summary = pd.DataFrame([{
            "symbol": symbol, "days": 0, "expected_days": expected_days,
            "complete_days": 0, "missing_days": expected_days, "incomplete_days": 0,
            "bars": 0, "expected_bars_if_full": expected_total,
            "missing_bar_estimate": expected_total, "completeness": 0.0,
        }])
        return pd.DataFrame(), summary

    x["date"] = x.ts.dt.strftime("%Y-%m-%d")
    daily = x.groupby("date").agg(
        bars=("ts", "count"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    cal = pd.DataFrame({"date": expected_dates})
    daily = cal.merge(daily, on="date", how="left")
    daily["bars"] = daily.bars.fillna(0).astype(int)
    daily["symbol"] = symbol
    daily["complete"] = daily.bars.between(46, 48)
    daily["bar_gap"] = (EXPECTED_BARS - daily.bars).clip(lower=0)

    summary = pd.DataFrame([{
        "symbol": symbol,
        "days": int((daily.bars > 0).sum()),
        "expected_days": expected_days,
        "complete_days": int(daily.complete.sum()),
        "missing_days": int((daily.bars == 0).sum()),
        "incomplete_days": int(((daily.bars > 0) & (~daily.complete)).sum()),
        "bars": int(daily.bars.sum()),
        "expected_bars_if_full": expected_total,
        "missing_bar_estimate": int(daily.bar_gap.sum()),
        "completeness": float(daily.bars.sum() / expected_total) if expected_total else 0.0,
    }])
    return daily, summary


def compare_sources(primary, secondary, symbol):
    a, b = normalize_a_share(primary), normalize_a_share(secondary)
    if a.empty or b.empty:
        return pd.DataFrame([{"symbol": symbol, "overlap": 0}])
    cols = ["ts", "open", "high", "low", "close", "volume"]
    z = a[cols].merge(b[cols], on="ts", suffixes=("_a", "_b"))
    if z.empty:
        return pd.DataFrame([{"symbol": symbol, "overlap": 0}])
    for c in ["open", "high", "low", "close"]:
        denom = z[f"{c}_a"].abs().replace(0, pd.NA)
        z[f"{c}_pct_diff"] = ((z[f"{c}_a"] - z[f"{c}_b"]).abs() / denom).astype(float)
    return pd.DataFrame([{
        "symbol": symbol,
        "overlap": len(z),
        "median_close_pct_diff": z.close_pct_diff.median(),
        "p99_close_pct_diff": z.close_pct_diff.quantile(.99),
        "max_close_pct_diff": z.close_pct_diff.max(),
    }])
