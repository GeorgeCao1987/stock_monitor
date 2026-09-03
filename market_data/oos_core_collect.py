from pathlib import Path
import pandas as pd

from config import START_DATE, END_DATE
from validate import validate_a_share

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

PCB_STOCKS = ["002916", "002463", "600183", "002938", "603228", "300476"]
INDEX = "000001.SH"


def save(df, path):
    if df is not None and not df.empty:
        df.to_csv(path, index=False)


def server_list():
    servers = [("180.153.18.170", 7709)]
    try:
        from pytdx.config.hosts import hq_hosts
        servers.extend((x[1], x[2]) for x in hq_hosts)
    except Exception:
        pass
    servers.extend([
        ("119.97.185.59", 7709), ("124.70.133.119", 7709),
        ("116.205.183.150", 7709), ("123.60.73.44", 7709),
        ("116.205.163.254", 7709), ("121.36.225.169", 7709),
        ("123.60.70.228", 7709), ("124.71.9.153", 7709),
        ("110.41.147.114", 7709), ("124.71.187.122", 7709),
    ])
    return list(dict.fromkeys(servers))


def normalize_tdx(df, source):
    if df is None or df.empty:
        return pd.DataFrame()
    z = df.copy()
    z["ts"] = pd.to_datetime(z.get("datetime"), errors="coerce")
    z = z.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts")
    z = z[(z.ts >= pd.Timestamp(START_DATE)) &
          (z.ts < pd.Timestamp(END_DATE) + pd.Timedelta(days=1))]
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        if col in z.columns:
            z[col] = pd.to_numeric(z[col], errors="coerce")
    return pd.DataFrame({
        "ts": z.ts,
        "open": z.open,
        "high": z.high,
        "low": z.low,
        "close": z.close,
        "volume": z["vol"] if "vol" in z.columns else 0,
        "amount": z["amount"] if "amount" in z.columns else None,
        "source": source,
    }).dropna(subset=["ts", "close"])


def expected_shape(df, symbol):
    if df is None or df.empty:
        return None, None
    _, sm = validate_a_share(df, symbol)
    r = sm.iloc[0]
    return r, sm


def is_exact(df, symbol):
    if df is None or df.empty:
        return False
    try:
        r, _ = expected_shape(df, symbol)
        expected_days = int(r["expected_days"])
        expected_bars = expected_days * 48
        return bool(
            expected_days > 0 and
            int(r["days"]) == expected_days and
            int(r["bars"]) == expected_bars and
            int(r["complete_days"]) == expected_days and
            int(r["missing_days"]) == 0 and
            int(r["incomplete_days"]) == 0 and
            float(r["completeness"]) == 1.0
        )
    except Exception:
        return False


def fetch_pytdx_history(symbol, is_index=False):
    from pytdx.hq import TdxHq_API

    code = "000001" if is_index else symbol
    market = 1 if (is_index or symbol.startswith("6")) else 0
    kind = "INDEX" if is_index else "STOCK"

    for ip, port in server_list():
        api = None
        try:
            api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
            if not api.connect(ip, port, time_out=2):
                continue

            getter = api.get_index_bars if is_index else api.get_security_bars
            test = getter(0, market, code, 0, 10) or []
            if len(test) < 5:
                print(f"{kind}_SERVER_BAD", symbol, ip, port, "test_rows", len(test))
                continue
            tdf = api.to_df(test)
            if "datetime" not in tdf.columns or not tdf["datetime"].notna().any():
                print(f"{kind}_SERVER_BAD", symbol, ip, port, "no_datetime")
                continue

            print(f"{kind}_SERVER_OK", symbol, ip, port,
                  "range", tdf.iloc[0]["datetime"], tdf.iloc[-1]["datetime"])
            parts = []
            for page in range(16):
                rows = getter(0, market, code, page * 800, 800) or []
                if not rows:
                    break
                x = api.to_df(rows)
                parts.append(x)
                ts = pd.to_datetime(x["datetime"], errors="coerce")
                print(f"{kind}_PAGE", symbol, page, len(x), ts.min(), ts.max())
                if ts.min() <= pd.Timestamp(START_DATE):
                    break

            if not parts:
                continue
            source = "pytdx_index" if is_index else "pytdx_stock"
            out = normalize_tdx(pd.concat(parts, ignore_index=True), source)
            if is_exact(out, symbol):
                print(f"{kind}_EXACT_OK", symbol, ip, port, "rows", len(out))
                return out
            print(f"{kind}_NOT_EXACT", symbol, ip, port, "rows", len(out))
        except Exception as e:
            print(f"{kind}_SERVER_FAIL", symbol, ip, port, repr(e))
        finally:
            if api is not None:
                try:
                    api.disconnect()
                except Exception:
                    pass

    raise RuntimeError(f"no TDX server returned exact history for {symbol} in {START_DATE}..{END_DATE}")


def assert_exact(df, symbol):
    r, sm = expected_shape(df, symbol)
    print("CORE_CHECK", symbol, r.to_dict())
    if not is_exact(df, symbol):
        raise RuntimeError(f"{symbol} failed exact session x 48 validation")
    return sm


def main():
    summaries = []

    for symbol in PCB_STOCKS:
        x = fetch_pytdx_history(symbol, is_index=False)
        sm = assert_exact(x, symbol)
        sm["primary_source"] = "pytdx_stock"
        summaries.append(sm)
        save(x.assign(symbol=symbol), DATA / f"a_{symbol}_pytdx_stock.csv")

    idx = fetch_pytdx_history(INDEX, is_index=True)
    sm = assert_exact(idx, INDEX)
    sm["primary_source"] = "pytdx_index"
    summaries.append(sm)
    save(idx.assign(symbol=INDEX), DATA / "a_000001_SH_pytdx_index.csv")

    report = pd.concat(summaries, ignore_index=True)
    report.to_csv(RESULTS / "completeness.csv", index=False)
    print("CORE_DATA_EXACT_OK", START_DATE, END_DATE)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
