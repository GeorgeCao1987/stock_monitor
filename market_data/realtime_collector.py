import json
import re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

from config import A_SHARES

OUT = Path(__file__).resolve().parent / "realtime"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent":"Mozilla/5.0", "Referer":"https://gu.qq.com/"}


def fetch_tencent_5m(code: str, count: int = 320) -> pd.DataFrame:
    url = "https://web.ifzq.gtimg.cn/appstock/app/kline/mkline"
    r = requests.get(url, params={"param":f"{code},m5,,{count}"}, headers=HEADERS, timeout=8)
    r.raise_for_status()
    txt = r.text.strip()
    if txt.startswith("var ") or "=" in txt[:80]:
        txt = txt.split("=",1)[1].rstrip(";\n ")
    js = json.loads(txt)
    data = (js.get("data") or {}).get(code) or {}
    bars = data.get("m5") or data.get("m5_data") or []
    rows=[]
    for p in bars:
        if len(p) < 6: continue
        rows.append({"ts":pd.to_datetime(p[0]),"open":float(p[1]),"close":float(p[2]),
                     "high":float(p[3]),"low":float(p[4]),"volume":float(p[5]),
                     "amount":float(p[6]) if len(p)>6 and p[6] not in (None,"") else None,
                     "source":"tencent"})
    return pd.DataFrame(rows)


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    manifest={"fetched_at_utc":stamp,"symbols":{}}
    for symbol,cfg in A_SHARES.items():
        code=cfg["sina"]
        try:
            df=fetch_tencent_5m(code)
            if not df.empty:
                df.to_csv(OUT/f"{symbol.replace('.','_')}_latest.csv",index=False)
            manifest["symbols"][symbol]={"rows":len(df),"source":"tencent","last_ts":str(df.ts.max()) if not df.empty else None}
        except Exception as e:
            manifest["symbols"][symbol]={"rows":0,"source":"tencent","error":repr(e)}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
