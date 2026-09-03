import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from config import A_SHARES, OVERSEAS, START_DATE, END_DATE

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def eastmoney_5m(symbol: str, secid: str) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "klt": 5,
        "fqt": 1,
        "beg": START_DATE.replace("-", ""),
        "end": END_DATE.replace("-", ""),
        "lmt": 100000,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    kl = (js.get("data") or {}).get("klines") or []
    rows = []
    for line in kl:
        x = line.split(",")
        rows.append({
            "ts": x[0], "open": float(x[1]), "close": float(x[2]),
            "high": float(x[3]), "low": float(x[4]), "volume": float(x[5]),
            "amount": float(x[6]), "source": "eastmoney", "symbol": symbol,
        })
    return pd.DataFrame(rows)


def sina_5m(symbol: str, sina_symbol: str, datalen: int = 2000) -> pd.DataFrame:
    url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
    params = {"symbol": sina_symbol, "scale": 5, "ma": "no", "datalen": datalen}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for x in js if isinstance(js, list) else []:
        day = x.get("day") or x.get("date")
        if not day or day[:10] < START_DATE or day[:10] > END_DATE:
            continue
        rows.append({
            "ts": day, "open": float(x["open"]), "close": float(x["close"]),
            "high": float(x["high"]), "low": float(x["low"]),
            "volume": float(x.get("volume", 0)), "amount": None,
            "source": "sina", "symbol": symbol,
        })
    return pd.DataFrame(rows)


def yahoo_5m(ticker: str, name: str) -> pd.DataFrame:
    # yfinance 5m history is suitable for recent windows; use explicit dates to avoid future leakage.
    df = yf.download(ticker, start=START_DATE, end=pd.Timestamp(END_DATE) + pd.Timedelta(days=1),
                     interval="5m", auto_adjust=False, progress=False, prepost=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[tcol], utc=True).dt.tz_convert("Asia/Shanghai").astype(str),
        "open": df["Open"].astype(float), "high": df["High"].astype(float),
        "low": df["Low"].astype(float), "close": df["Close"].astype(float),
        "volume": df.get("Volume", pd.Series([0]*len(df))).astype(float),
        "amount": None, "source": "yahoo", "symbol": ticker, "name": name,
    })
    return out


def save(df: pd.DataFrame, name: str):
    if df.empty:
        print(f"WARN empty: {name}")
        return
    df = df.sort_values("ts").drop_duplicates(["symbol", "ts", "source"])
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(df), df.iloc[0]["ts"], df.iloc[-1]["ts"])


def main():
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    for symbol, cfg in A_SHARES.items():
        try:
            em = eastmoney_5m(symbol, cfg["secid"])
            save(em, f"a_{symbol.replace('.', '_')}_eastmoney")
            manifest["files"][f"a_{symbol}_eastmoney"] = len(em)
        except Exception as e:
            print("EASTMONEY ERROR", symbol, repr(e))
        time.sleep(0.3)
        try:
            sn = sina_5m(symbol, cfg["sina"])
            save(sn, f"a_{symbol.replace('.', '_')}_sina")
            manifest["files"][f"a_{symbol}_sina"] = len(sn)
        except Exception as e:
            print("SINA ERROR", symbol, repr(e))
    for ticker, name in OVERSEAS.items():
        try:
            df = yahoo_5m(ticker, name)
            save(df, "o_" + ticker.replace("^", "IDX_").replace("=", "_").replace(".", "_"))
            manifest["files"][f"o_{ticker}"] = len(df)
        except Exception as e:
            print("YAHOO ERROR", ticker, repr(e))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
