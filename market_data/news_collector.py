import json
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd

from config import START_DATE, END_DATE

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}

QUERIES = [
    '(NVIDIA OR "SK Hynix" OR Samsung OR semiconductor OR AI chip OR export control)',
    '(oil OR crude) AND (Treasury OR bond yield OR inflation)',
    '(China AND (PCB OR server OR AI hardware OR semiconductor))',
]
NEGATIVE = ["ban", "restriction", "sanction", "war", "attack", "inflation", "yield surge", "plunge", "selloff", "tariff"]
POSITIVE = ["beat", "record", "upgrade", "stimulus", "cut rates", "orders", "capex", "rally", "surge"]


def gdelt(query):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "sort": "datedesc",
        "startdatetime": START_DATE.replace("-", "") + "000000",
        "enddatetime": END_DATE.replace("-", "") + "235959",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return (r.json() or {}).get("articles", [])


def score_title(title):
    t = (title or "").lower()
    neg = sum(k in t for k in NEGATIVE)
    pos = sum(k in t for k in POSITIVE)
    return pos - neg


def main():
    rows = []
    for q in QUERIES:
        try:
            for a in gdelt(q):
                seen = a.get("seendate")
                ts = pd.to_datetime(seen, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                rows.append({
                    "published_ts": ts.tz_convert("Asia/Shanghai").isoformat(),
                    "title": a.get("title"), "url": a.get("url"),
                    "domain": a.get("domain"), "language": a.get("language"),
                    "query": q, "event_score": score_title(a.get("title")),
                    "source": "gdelt",
                })
        except Exception as e:
            print("GDELT ERROR", q, repr(e))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published_ts").drop_duplicates(["url", "published_ts"])
        df.to_csv(OUT / "news_events.csv", index=False)
    print("news rows", len(df))


if __name__ == "__main__":
    main()
