import os

START_DATE = os.getenv("MARKET_START_DATE", "2026-07-01")
END_DATE = os.getenv("MARKET_END_DATE", "2026-08-31")
TZ = "Asia/Shanghai"

A_SHARES = {
    "002916": {"name": "深南电路", "secid": "0.002916", "sina": "sz002916", "yahoo": "002916.SZ"},
    "002463": {"name": "沪电股份", "secid": "0.002463", "sina": "sz002463", "yahoo": "002463.SZ"},
    "600183": {"name": "生益科技", "secid": "1.600183", "sina": "sh600183", "yahoo": "600183.SS"},
    "002938": {"name": "鹏鼎控股", "secid": "0.002938", "sina": "sz002938", "yahoo": "002938.SZ"},
    "603228": {"name": "景旺电子", "secid": "1.603228", "sina": "sh603228", "yahoo": "603228.SS"},
    "300476": {"name": "胜宏科技", "secid": "0.300476", "sina": "sz300476", "yahoo": "300476.SZ"},
    "000001.SH": {"name": "上证指数", "secid": "1.000001", "sina": "sh000001", "yahoo": "000001.SS"},
}

OVERSEAS = {
    "^KS11": "KOSPI",
    "005930.KS": "三星电子",
    "000660.KS": "SK海力士",
    "^SOX": "SOX",
    "NQ=F": "纳指期货",
    "CL=F": "WTI",
    "GC=F": "黄金",
    "^TYX": "美国30Y收益率",
}

PCB_MEMBERS = ["002916", "002463", "600183", "002938", "603228", "300476"]
