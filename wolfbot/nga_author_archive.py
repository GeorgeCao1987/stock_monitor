from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
KNOWLEDGE = ROOT / "knowledge"
RAW_ROOT = KNOWLEDGE / "raw_sources"
DERIVED_ROOT = KNOWLEDGE / "derived"
RAW_INDEX = KNOWLEDGE / "raw_source_index.jsonl"

UID = int(os.getenv("NGA_UID", "150058"))
TID = int(os.getenv("NGA_TID", "47288722"))
BASE = os.getenv("NGA_BASE_URL", "https://bbs.nga.cn").rstrip("/")
MAX_PAGES = int(os.getenv("NGA_ARCHIVE_MAX_PAGES", "5000"))
SLEEP = float(os.getenv("NGA_ARCHIVE_SLEEP", "0.8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_ts(value: str) -> str:
    s = re.sub(r"[^0-9]", "", value or "")
    return s[:14] if s else "unknown"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_cookie(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if raw.lower().startswith("cookie:"):
        raw = raw.split(":", 1)[1].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k.strip():
            out[k.strip()] = v.strip()
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def decode_json_response(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8-sig", errors="replace")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1], strict=False)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    payload = obj.get("data")
    return payload if isinstance(payload, dict) else obj


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    r = payload.get("__R", [])
    if isinstance(r, list):
        return [x for x in r if isinstance(x, dict)]
    if isinstance(r, dict):
        def key_order(item: tuple[str, Any]) -> tuple[int, str]:
            k = str(item[0])
            return (int(k) if k.isdigit() else 10**9, k)
        return [v for _, v in sorted(r.items(), key=key_order) if isinstance(v, dict)]
    return []


def main() -> None:
    observed_at = now_iso()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": os.getenv(
                "NGA_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36",
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": BASE + "/",
        }
    )
    cookie_map = parse_cookie(os.getenv("NGA_COOKIE", ""))
    if cookie_map:
        session.cookies.update(cookie_map)

    old_index = load_jsonl(RAW_INDEX)
    seen_triplets = {
        (str(x.get("tid", "")), str(x.get("pid", "")), str(x.get("content_sha256", "")))
        for x in old_index
    }
    manifests = list(old_index)
    corpus: list[dict[str, Any]] = []
    signatures: set[tuple[str, ...]] = set()
    raw_dir = RAW_ROOT / f"nga_tid{TID}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages_fetched = 0
    added = 0

    for page in range(1, MAX_PAGES + 1):
        response = session.get(
            BASE + "/read.php",
            params={"tid": TID, "authorid": UID, "page": page, "__output": 11},
            timeout=25,
        )
        response.raise_for_status()
        payload = decode_json_response(response.content)
        rows = [x for x in rows_from_payload(payload) if str(x.get("authorid", "")) == str(UID)]
        signature = tuple(str(x.get("pid", "")) + ":" + str(x.get("lou", "")) for x in rows)
        if not rows or signature in signatures:
            break
        signatures.add(signature)
        pages_fetched += 1

        for row in rows:
            raw_content = row.get("content", "")
            if raw_content is None:
                raw_content = ""
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)
            content_bytes = raw_content.encode("utf-8")
            content_sha = sha256_bytes(content_bytes)
            pid = str(row.get("pid", "0"))
            floor = str(row.get("lou", ""))
            posted_at = str(row.get("postdate", ""))
            source_key = f"nga:{TID}:{pid}:{content_sha}"
            triplet = (str(TID), pid, content_sha)

            base_name = f"{safe_ts(posted_at)}_pid{pid}_floor{floor}"
            raw_path = raw_dir / f"{base_name}.txt"
            if triplet not in seen_triplets:
                if raw_path.exists():
                    raw_path = raw_dir / f"{base_name}_observed_{safe_ts(observed_at)}.txt"
                raw_path.write_bytes(content_bytes)
                manifest = {
                    "source_id": source_key,
                    "source_type": "nga_author_only_post",
                    "uid": UID,
                    "tid": TID,
                    "pid": pid,
                    "floor": floor,
                    "page": page,
                    "author": "-阿狼-",
                    "source_posted_at": posted_at,
                    "observed_at": observed_at,
                    "url": f"{BASE}/read.php?tid={TID}&authorid={UID}&page={page}",
                    "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
                    "content_sha256": content_sha,
                    "content_bytes": len(content_bytes),
                    "immutable": True,
                }
                manifests.append(manifest)
                seen_triplets.add(triplet)
                added += 1

            corpus.append(
                {
                    "uid": UID,
                    "tid": TID,
                    "pid": pid,
                    "floor": floor,
                    "page": page,
                    "posted_at": posted_at,
                    "url": f"{BASE}/read.php?tid={TID}&authorid={UID}&page={page}",
                    "content_sha256": content_sha,
                    "content": raw_content,
                }
            )
        time.sleep(SLEEP)

    # Immutable raw files are never rewritten. The index is append-only by (tid,pid,hash).
    if added:
        write_jsonl(RAW_INDEX, manifests)

    # This is a derived convenience view and may be regenerated. The `content` field is copied
    # byte-for-byte at the Unicode-text level from the API response; raw canonical files remain above.
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    review_path = DERIVED_ROOT / f"nga_tid{TID}_review.jsonl"
    write_jsonl(review_path, corpus)

    state = {
        "uid": UID,
        "tid": TID,
        "observed_at": observed_at,
        "pages_fetched": pages_fetched,
        "posts_in_current_view": len(corpus),
        "new_immutable_sources": added,
        "earliest_posted_at": min((x.get("posted_at", "") for x in corpus), default=""),
        "latest_posted_at": max((x.get("posted_at", "") for x in corpus), default=""),
        "review_path": str(review_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_root": str(raw_dir.relative_to(ROOT)).replace("\\", "/"),
    }
    (DERIVED_ROOT / f"nga_tid{TID}_archive_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
