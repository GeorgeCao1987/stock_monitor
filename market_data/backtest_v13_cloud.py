from pathlib import Path
import pandas as pd
import backtest_v13 as bt

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"


def preferred_source(symbol):
    p = RESULTS / "completeness.csv"
    if not p.exists():
        return None
    try:
        sm = pd.read_csv(p, dtype={"symbol": str})
        row = sm[sm["symbol"].astype(str) == str(symbol)]
        if row.empty or "primary_source" not in row.columns:
            return None
        src = str(row.iloc[0]["primary_source"]).strip()
        return src if src and src.lower() != "nan" else None
    except Exception:
        return None


def load_a_cloud(symbol, source=None):
    key = symbol.replace('.', '_')
    preferred = source or preferred_source(symbol)
    order = [preferred, "baostock", "mootdx", "sina", "eastmoney", "yahoo"]
    seen = set()
    for src in order:
        if not src or src in seen:
            continue
        seen.add(src)
        p = DATA / f"a_{key}_{src}.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p)
        if x.empty:
            continue
        x["ts"] = pd.to_datetime(x["ts"])
        if getattr(x["ts"].dt, "tz", None) is not None:
            x["ts"] = x["ts"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        return x.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return pd.DataFrame()


if __name__ == "__main__":
    bt.load_a = load_a_cloud
    bt.main()
