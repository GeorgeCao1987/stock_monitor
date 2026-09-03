import socket
import pandas as pd


def _query_logged_in(bs, symbol, start, end):
    market = "sh" if symbol.startswith("sh") else "sz"
    code = symbol[-6:]
    bs_code = f"{market}.{code}"
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,time,code,open,high,low,close,volume,amount,adjustflag",
        start_date=start, end_date=end, frequency="5", adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock query {bs_code} {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        p = dict(zip(rs.fields, rs.get_row_data()))
        ts = pd.to_datetime(
            (p.get("time") or "")[:14],
            format="%Y%m%d%H%M%S", errors="coerce",
        )
        if pd.isna(ts):
            continue
        rows.append({
            "ts": ts,
            "open": float(p["open"]),
            "high": float(p["high"]),
            "low": float(p["low"]),
            "close": float(p["close"]),
            "volume": float(p["volume"] or 0),
            "amount": float(p["amount"] or 0),
            "source": "baostock",
        })
    return pd.DataFrame(rows)


def fetch_baostock_batch(symbol_map, start, end):
    """Fetch all A-share/index 5m histories under one Baostock login.

    symbol_map maps internal symbols (e.g. 002916, 000001.SH) to exchange-prefixed
    quote symbols (e.g. sz002916, sh000001). A failed individual query becomes an
    empty frame so the caller can invoke its fallback source without discarding the
    other successful symbols.
    """
    import baostock as bs

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(12)
    out = {key: pd.DataFrame() for key in symbol_map}
    try:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login {lg.error_code}: {lg.error_msg}")
        try:
            for key, quote_symbol in symbol_map.items():
                try:
                    x = _query_logged_in(bs, quote_symbol, start, end)
                    out[key] = x
                    print("BAOSTOCK_BATCH", key, "rows", len(x))
                except Exception as e:
                    print("BAOSTOCK_BATCH", key, "FAILED", repr(e))
        finally:
            bs.logout()
    finally:
        socket.setdefaulttimeout(old)
    return out
