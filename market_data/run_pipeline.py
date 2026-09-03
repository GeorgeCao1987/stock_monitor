from pathlib import Path
import time
import pandas as pd

from config import A_SHARES, START_DATE, END_DATE, OVERSEAS
from collectors import (
    fetch_mootdx_5m, fetch_sina_5m,
    fetch_yahoo_recent_5m, fetch_yahoo_daily,
)
from baostock_batch import fetch_baostock_batch
from validate import validate_a_share, compare_sources

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def save(df, path):
    if df is not None and not df.empty:
        df.to_csv(path, index=False)


def safe(call, label, attempts=1, require_nonempty=False):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            x = call()
            if require_nonempty and (x is None or x.empty):
                raise RuntimeError("empty response")
            return x
        except Exception as e:
            last = e
            print(label, f"FAILED attempt {attempt}/{attempts}", repr(e))
            if attempt < attempts:
                time.sleep(2 * attempt)
    print(label, "GIVE_UP", repr(last))
    return pd.DataFrame()


def trim(x):
    if x is None or x.empty:
        return pd.DataFrame()
    x = x.copy()
    x["ts"] = pd.to_datetime(x.ts)
    return x[
        (x.ts >= pd.Timestamp(START_DATE)) &
        (x.ts < pd.Timestamp(END_DATE) + pd.Timedelta(days=1))
    ].copy()


def complete_enough(x, symbol):
    if x is None or x.empty:
        return False
    _, sm = validate_a_share(x, symbol)
    r = sm.iloc[0]
    return bool(
        float(r["completeness"]) >= .999 and
        int(r["missing_days"]) == 0 and
        int(r["incomplete_days"]) == 0
    )


def main():
    summaries = []
    comparisons = []

    # One Baostock login for all seven histories. If the login itself fails,
    # fallbacks remain available on a per-symbol basis.
    try:
        bs_map = fetch_baostock_batch(
            {symbol: cfg["sina"] for symbol, cfg in A_SHARES.items()},
            START_DATE, END_DATE,
        )
    except Exception as e:
        print("BAOSTOCK_BATCH_LOGIN_FAILED", repr(e))
        bs_map = {symbol: pd.DataFrame() for symbol in A_SHARES}

    for symbol, cfg in A_SHARES.items():
        bs = trim(bs_map.get(symbol, pd.DataFrame()))

        # Only fall back to TDX when Baostock is not sufficiently complete.
        td = pd.DataFrame()
        if not complete_enough(bs, symbol):
            td = trim(safe(
                lambda: fetch_mootdx_5m(
                    cfg["sina"], START_DATE, END_DATE,
                    is_index=(symbol == "000001.SH")
                ),
                f"mootdx {symbol}", attempts=2, require_nonempty=True,
            ))

        # Sina is a recent-history verifier, never an older-history backtest fallback.
        sn = trim(safe(lambda: fetch_sina_5m(cfg["sina"], 1023), f"sina {symbol}"))
        if not sn.empty and ("amount" not in sn.columns or sn.amount.isna().all()):
            sn["amount"] = sn.close * sn.volume

        save(bs.assign(symbol=symbol), DATA / f"a_{symbol.replace('.', '_')}_baostock.csv")
        save(td.assign(symbol=symbol), DATA / f"a_{symbol.replace('.', '_')}_mootdx.csv")
        save(sn.assign(symbol=symbol), DATA / f"a_{symbol.replace('.', '_')}_sina.csv")

        if complete_enough(bs, symbol):
            primary, source = bs, "baostock"
        elif complete_enough(td, symbol):
            primary, source = td, "mootdx"
        elif not bs.empty:
            primary, source = bs, "baostock"
        elif not td.empty:
            primary, source = td, "mootdx"
        else:
            primary, source = sn, "sina"

        daily, summary = validate_a_share(primary, symbol)
        if not summary.empty:
            summary["primary_source"] = source
        save(daily, RESULTS / f"daily_{symbol.replace('.', '_')}.csv")
        summaries.append(summary)

        for a, b, pair in [
            (bs, td, "baostock-mootdx"),
            (bs, sn, "baostock-sina"),
            (td, sn, "mootdx-sina"),
        ]:
            if not a.empty and not b.empty:
                c = compare_sources(a, b, symbol)
                c["pair"] = pair
                comparisons.append(c)

    for ticker, name in OVERSEAS.items():
        x = safe(
            lambda t=ticker: fetch_yahoo_recent_5m(t, START_DATE, END_DATE),
            f"overseas {ticker}", attempts=2,
        )
        if not x.empty:
            x["symbol"], x["name"] = ticker, name
        save(x, DATA / ("o_" + ticker.replace("^", "IDX_").replace("=", "_").replace(".", "_") + ".csv"))

    macro_start = (pd.Timestamp(START_DATE) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    for ticker, label in [("^TYX", "tyx"), ("CL=F", "oil"), ("^SOX", "sox")]:
        d = safe(
            lambda t=ticker: fetch_yahoo_daily(t, macro_start, END_DATE),
            f"daily macro {ticker}", attempts=2,
        )
        save(d, DATA / f"macro_{label}_daily_3y.csv")

    sm = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    cp = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    save(sm, RESULTS / "completeness.csv")
    save(cp, RESULTS / "source_compare.csv")
    print("COMPLETENESS")
    print(sm.to_string(index=False) if not sm.empty else "none")
    print("SOURCE_COMPARE")
    print(cp.to_string(index=False) if not cp.empty else "none")

    if sm.empty or "completeness" not in sm:
        raise SystemExit("missing completeness report")
    bad = sm[
        (sm["completeness"] < .999) |
        (sm["missing_days"] > 0) |
        (sm["incomplete_days"] > 0)
    ]
    if not bad.empty:
        print("STRICT_COMPLETENESS_FAILURE")
        print(bad.to_string(index=False))
        raise SystemExit("A-share/index data not sufficiently complete for backtest")


if __name__ == "__main__":
    main()
