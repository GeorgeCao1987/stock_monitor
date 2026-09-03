from pathlib import Path
import json
import pandas as pd

from config import START_DATE, END_DATE, A_SHARES, OVERSEAS
from collectors import fetch_eastmoney_5m, fetch_sina_5m, fetch_yahoo_5m
from validate import validate_a_share, compare_sources

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

summaries = []
source_checks = []
errors = []

for symbol, meta in A_SHARES.items():
    try:
        em = fetch_eastmoney_5m(meta["secid"], START_DATE, END_DATE)
        em["symbol"] = symbol
        em.to_parquet(OUT / f"{symbol.replace('.', '_')}_eastmoney_5m.parquet", index=False)
        daily, summary = validate_a_share(em, symbol)
        daily.to_csv(OUT / f"{symbol.replace('.', '_')}_daily_validation.csv", index=False)
        summaries.append(summary)
        try:
            sina = fetch_sina_5m(meta["sina"], 2000)
            sina["symbol"] = symbol
            sina.to_parquet(OUT / f"{symbol.replace('.', '_')}_sina_5m.parquet", index=False)
            source_checks.append(compare_sources(em, sina, symbol))
        except Exception as e:
            errors.append({"symbol": symbol, "source": "sina", "error": repr(e)})
    except Exception as e:
        errors.append({"symbol": symbol, "source": "eastmoney", "error": repr(e)})

for ticker, name in OVERSEAS.items():
    try:
        df = fetch_yahoo_5m(ticker, START_DATE, END_DATE)
        df["symbol"] = ticker
        df["name"] = name
        df.to_parquet(OUT / f"overseas_{ticker.replace('^','idx_').replace('=','_').replace('.','_')}_5m.parquet", index=False)
    except Exception as e:
        errors.append({"symbol": ticker, "source": "yahoo", "error": repr(e)})

if summaries:
    pd.concat(summaries, ignore_index=True).to_csv(OUT / "a_share_completeness.csv", index=False)
if source_checks:
    pd.concat(source_checks, ignore_index=True).to_csv(OUT / "a_share_source_compare.csv", index=False)

(OUT / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

print("=== A-share completeness ===")
if summaries:
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))
print("=== source compare ===")
if source_checks:
    print(pd.concat(source_checks, ignore_index=True).to_string(index=False))
print("=== errors ===")
print(json.dumps(errors, ensure_ascii=False, indent=2))
