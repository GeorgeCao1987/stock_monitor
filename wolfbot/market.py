from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass
class MarketSnapshot:
    code: str
    name: str
    timestamp: str
    price: float | None
    pct: float | None
    open: float | None
    high: float | None
    low: float | None
    prev_close: float | None
    amount: float | None
    volume: float | None
    ma5: float | None
    ma10: float | None
    ma13: float | None
    ma20: float | None
    ma60: float | None
    ma144: float | None
    day20_high: float | None
    day20_low: float | None
    amount_ratio_same_time: float | None
    trend_hint: str
    source: str = "eastmoney"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_code(text: str) -> str | None:
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return m.group(1) if m else None


def secid(code: str) -> str:
    # Main-board convention. Shanghai 6/9/5; Shenzhen 0/2/3.
    market = "1" if code.startswith(("5", "6", "9")) else "0"
    return f"{market}.{code}"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_quote(code: str) -> dict[str, Any]:
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    fields = "f57,f58,f43,f44,f45,f46,f47,f48,f60,f170"
    r = requests.get(url, params={"secid": secid(code), "fields": fields}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    d = (r.json().get("data") or {})

    def price_field(key):
        v = _f(d.get(key))
        return None if v is None else v / 100.0

    return {
        "code": str(d.get("f57") or code),
        "name": str(d.get("f58") or ""),
        "price": price_field("f43"),
        "high": price_field("f44"),
        "low": price_field("f45"),
        "open": price_field("f46"),
        "volume": _f(d.get("f47")),
        "amount": _f(d.get("f48")),
        "prev_close": price_field("f60"),
        "pct": None if _f(d.get("f170")) is None else _f(d.get("f170")) / 100.0,
    }


def fetch_daily(code: str, limit: int = 220) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid(code),
        "klt": 101,
        "fqt": 1,
        "lmt": limit,
        "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    lines = ((r.json().get("data") or {}).get("klines") or [])
    rows = []
    for line in lines:
        x = line.split(",")
        if len(x) < 7:
            continue
        rows.append(
            {
                "date": x[0],
                "open": float(x[1]),
                "close": float(x[2]),
                "high": float(x[3]),
                "low": float(x[4]),
                "volume": float(x[5]),
                "amount": float(x[6]),
            }
        )
    return pd.DataFrame(rows)


def fetch_5m(code: str, limit: int = 1200) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid(code),
        "klt": 5,
        "fqt": 1,
        "lmt": limit,
        "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    lines = ((r.json().get("data") or {}).get("klines") or [])
    rows = []
    for line in lines:
        x = line.split(",")
        if len(x) < 7:
            continue
        rows.append(
            {
                "ts": pd.to_datetime(x[0]),
                "open": float(x[1]),
                "close": float(x[2]),
                "high": float(x[3]),
                "low": float(x[4]),
                "volume": float(x[5]),
                "amount": float(x[6]),
            }
        )
    return pd.DataFrame(rows)


def _same_time_amount_ratio(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    df = df.copy()
    df["date"] = df["ts"].dt.date
    days = sorted(df["date"].unique())
    if len(days) < 2:
        return None
    today, prev = days[-1], days[-2]
    a = df[df["date"] == today]
    b = df[df["date"] == prev]
    if a.empty or b.empty:
        return None
    n = min(len(a), len(b))
    if n <= 0:
        return None
    today_amt = float(a.iloc[:n]["amount"].sum())
    prev_amt = float(b.iloc[:n]["amount"].sum())
    return None if prev_amt <= 0 else today_amt / prev_amt


def build_snapshot(code: str) -> MarketSnapshot:
    q = fetch_quote(code)
    daily = fetch_daily(code)
    m5 = fetch_5m(code)

    close = daily["close"] if not daily.empty else pd.Series(dtype=float)
    mas: dict[int, float | None] = {}
    for n in (5, 10, 13, 20, 60, 144):
        mas[n] = float(close.tail(n).mean()) if len(close) >= n else None

    high20 = float(daily["high"].tail(20).max()) if len(daily) >= 1 else None
    low20 = float(daily["low"].tail(20).min()) if len(daily) >= 1 else None
    price = q.get("price")

    if price is None:
        trend = "unknown"
    elif mas[144] is not None and price < mas[144]:
        trend = "below_ma144"
    elif mas[20] is not None and mas[60] is not None and price > mas[20] > mas[60]:
        trend = "uptrend"
    elif mas[20] is not None and price < mas[20]:
        trend = "adjustment_or_downtrend"
    else:
        trend = "neutral"

    return MarketSnapshot(
        code=code,
        name=q.get("name", ""),
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        price=price,
        pct=q.get("pct"),
        open=q.get("open"),
        high=q.get("high"),
        low=q.get("low"),
        prev_close=q.get("prev_close"),
        amount=q.get("amount"),
        volume=q.get("volume"),
        ma5=mas[5],
        ma10=mas[10],
        ma13=mas[13],
        ma20=mas[20],
        ma60=mas[60],
        ma144=mas[144],
        day20_high=high20,
        day20_low=low20,
        amount_ratio_same_time=_same_time_amount_ratio(m5),
        trend_hint=trend,
    )
