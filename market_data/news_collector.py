from pathlib import Path
import requests
import pandas as pd

from config import START_DATE, END_DATE

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}

# External-message layer. These queries intentionally cover information that can
# move A-share AI/PCB risk appetite before or during the session, and are kept
# separate from overseas price/index priors.
QUERY_SPECS = [
    {
        "query": '(Federal Reserve OR US jobs OR CPI OR PPI OR Treasury yield OR dollar OR tariff OR export control)',
        "scope": "GLOBAL",
        "chain": "macro;risk_appetite;semiconductor",
    },
    {
        "query": '(NVIDIA OR Broadcom OR AMD OR Micron OR TSMC OR "AI server" OR hyperscaler OR capex) AND (AI OR semiconductor OR datacenter)',
        "scope": "AI_SEMI",
        "chain": "AI;semiconductor;server;PCB;CPO",
    },
    {
        "query": '(Samsung OR "SK Hynix" OR KOSPI OR Nikkei OR "Tokyo Electron" OR Advantest) AND (chip OR semiconductor OR memory OR AI)',
        "scope": "ASIA",
        "chain": "memory;semiconductor;AI_hardware",
    },
    {
        "query": '(China AND (PCB OR server OR AI hardware OR semiconductor OR export control OR data center))',
        "scope": "CHINA",
        "chain": "PCB;AI_hardware;server;semiconductor",
    },
    {
        "query": '(war OR attack OR sanction OR shipping disruption OR Strait OR missile) AND (Asia OR China OR oil OR semiconductor)',
        "scope": "GLOBAL",
        "chain": "geopolitics;oil;risk_appetite;semiconductor",
    },
]

NEGATIVE = [
    "ban", "restriction", "sanction", "war", "attack", "missile", "inflation",
    "yield surge", "plunge", "selloff", "tariff", "export control", "delay",
    "cut forecast", "downgrade", "shortage", "disruption", "probe", "investigation",
]
POSITIVE = [
    "beat", "record", "upgrade", "stimulus", "rate cut", "cut rates", "orders",
    "capex", "rally", "surge", "raise forecast", "approval", "expand", "accelerate",
    "strong demand", "new high", "investment", "buyback",
]


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


def direction_from_score(score: int) -> int:
    return 1 if score > 0 else (-1 if score < 0 else 0)


def strength_from_score(score: int) -> float:
    if score == 0:
        return 0.0
    return min(1.0, 0.45 + 0.18 * abs(score))


def main():
    rows = []
    for spec in QUERY_SPECS:
        q = spec["query"]
        try:
            for a in gdelt(q):
                seen = a.get("seendate")
                ts = pd.to_datetime(seen, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                score = score_title(a.get("title"))
                rows.append({
                    "published_ts": ts.tz_convert("Asia/Shanghai").isoformat(),
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "query": q,
                    "source": "gdelt",
                    "raw_title_score": score,
                    "event_direction": direction_from_score(score),
                    "event_strength": strength_from_score(score),
                    "event_scope": spec["scope"],
                    "event_freshness": 1.0,  # recomputed at each decision timestamp
                    "event_confidence": 0.65,
                    "affected_chain": spec["chain"],
                    "event_source_count": 1,
                    "event_active": True,
                })
        except Exception as e:
            print("GDELT ERROR", q, repr(e))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published_ts").drop_duplicates(["url", "published_ts"])
        df.to_csv(OUT / "news_events.csv", index=False)
    print("external news rows", len(df))
    if not df.empty:
        print(df[["published_ts", "event_direction", "event_strength", "event_scope", "title"]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
