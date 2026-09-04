from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE = "https://bbs.nga.cn"
DEFAULT_BASES = (
    "https://bbs.nga.cn",
    "https://nga.178.com",
    "https://bbs.ngacn.cc",
)
DEFAULT_UID = 150058
DEFAULT_TID = 45974302
DEFAULT_SLEEP = 1.2


@dataclass
class NgaPost:
    uid: int
    tid: int
    page: int
    floor: str
    pid: str
    author: str
    posted_at: str
    content: str
    url: str
    source: str = "nga"


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\t\r ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_pid(tid: int, page: int, floor: str, content: str) -> str:
    raw = f"{tid}|{page}|{floor}|{content}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:20]


def _parse_cookie_string(cookie: str) -> dict[str, str]:
    """Parse a browser Cookie header value robustly.

    Users sometimes paste either the raw value or the whole `Cookie: ...` line,
    and sometimes GitHub Secrets keep surrounding quotes. Accept all of these.
    """
    raw = (cookie or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1].strip()
    if raw.lower().startswith("cookie:"):
        raw = raw.split(":", 1)[1].strip()

    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip()
    return out


def _decode_bytes(data: bytes, preferred: str | None = None) -> str:
    encodings: list[str] = []
    if preferred:
        encodings.append(preferred)
    encodings.extend(["utf-8-sig", "utf-8", "gb18030"])
    seen: set[str] = set()
    for enc in encodings:
        enc = enc.lower()
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def _epoch_or_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        try:
            n = int(value)
            if n > 1_000_000_000:
                return datetime.fromtimestamp(n, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return str(value)


class NgaCrawler:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        cookie: str | None = None,
        sleep_seconds: float = DEFAULT_SLEEP,
        timeout: int = 20,
    ) -> None:
        configured = [x.strip().rstrip("/") for x in os.getenv("NGA_BASE_URLS", "").split(",") if x.strip()]
        candidates = configured or [base_url.rstrip("/"), *DEFAULT_BASES]
        self.base_urls: list[str] = []
        for x in candidates:
            if x and x not in self.base_urls:
                self.base_urls.append(x)
        self.base_url = self.base_urls[0]
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.getenv("NGA_USER_AGENT", "Nga_Official/80023"),
                "X-User-Agent": "Nga_Official",
                "Accept-Language": "zh-CN,zh;q=0.9",
            }
        )
        cookie = cookie or os.getenv("NGA_COOKIE", "")
        self.cookie_map = _parse_cookie_string(cookie)
        if self.cookie_map:
            self.session.cookies.update(self.cookie_map)

    @staticmethod
    def _looks_blocked(text: str) -> bool:
        markers = [
            "版面关闭",
            "你可能需要",
            "登录后访问",
            "你必须登录",
            "ERROR:2048",
            "权限不足",
        ]
        return any(x in text for x in markers)

    def _get(self, base: str, path: str, params: dict[str, Any]) -> requests.Response:
        url = urljoin(base + "/", path.lstrip("/"))
        headers = {"Referer": base + "/"}
        r = self.session.get(url, params=params, timeout=self.timeout, headers=headers)
        r.raise_for_status()
        return r

    def _parse_api_page(self, raw: bytes, uid: int, tid: int, page: int, base: str) -> tuple[list[NgaPost], bool]:
        text = _decode_bytes(raw, "utf-8")
        if self._looks_blocked(text):
            return [], True

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return [], False
        try:
            data = json.loads(text[start : end + 1], strict=False)
        except Exception:
            return [], False
        if not isinstance(data, dict):
            return [], False

        serialized = json.dumps(data, ensure_ascii=False)
        if self._looks_blocked(serialized):
            return [], True

        replies = data.get("__R", {})
        users = data.get("__U", {})
        if isinstance(replies, list):
            reply_rows = replies
        elif isinstance(replies, dict):
            def _sort_key(item: tuple[str, Any]) -> tuple[int, str]:
                k = str(item[0])
                return (int(k) if k.isdigit() else 10**9, k)
            reply_rows = [v for _, v in sorted(replies.items(), key=_sort_key)]
        else:
            reply_rows = []

        posts: list[NgaPost] = []
        for row in reply_rows:
            if not isinstance(row, dict):
                continue
            try:
                author_id = int(row.get("authorid", 0))
            except Exception:
                author_id = 0
            if author_id != uid:
                continue
            content = row.get("content")
            if content is None:
                continue
            content = _clean_text(str(content))
            if not content:
                continue

            pid = str(row.get("pid") or "")
            floor = str(row.get("lou") if row.get("lou") is not None else "")
            if not pid:
                pid = _make_pid(tid, page, floor, content)

            author = "[-阿狼-]"
            if isinstance(users, dict):
                u = users.get(str(uid)) or users.get(uid)
                if isinstance(u, dict) and u.get("username"):
                    author = _clean_text(str(u.get("username")))

            posts.append(
                NgaPost(
                    uid=uid,
                    tid=tid,
                    page=page,
                    floor=floor,
                    pid=pid,
                    author=author,
                    posted_at=_epoch_or_text(row.get("postdate")),
                    content=content,
                    url=f"{base}/read.php?tid={tid}&authorid={uid}&page={page}",
                )
            )
        return posts, False

    def _parse_author_only_page(
        self,
        html: str,
        uid: int,
        tid: int,
        page: int,
        base: str,
    ) -> list[NgaPost]:
        soup = BeautifulSoup(html, "lxml")
        posts: list[NgaPost] = []

        rows = soup.select("[id^='postrow']")
        if not rows:
            content_nodes = soup.select("[id^='postcontent']")
            rows = [n.parent for n in content_nodes if n.parent is not None]

        for row in rows:
            author_link = row.select_one("a[href*='func=ucp'][href*='uid=']")
            if author_link:
                href = author_link.get("href", "")
                m_uid = re.search(r"uid=(\d+)", href)
                if m_uid and int(m_uid.group(1)) != uid:
                    continue
                author = _clean_text(author_link.get_text(" ", strip=True))
            else:
                author = ""

            content_node = row.select_one("[id^='postcontent']") or row.select_one(".postcontent")
            if not content_node:
                continue
            # Remove rendered quote blocks where possible so another user's words do not
            # become candidate rules attributed to 阿狼.
            for q in content_node.select(".quote, blockquote, [class*='quote']"):
                q.decompose()
            content = _clean_text(content_node.get_text("\n", strip=True))
            if not content:
                continue

            row_id = row.get("id", "")
            floor_match = re.search(r"(\d+)$", row_id)
            floor = floor_match.group(1) if floor_match else ""

            pid = ""
            for a in row.select("a[href]"):
                href = a.get("href", "")
                m_pid = re.search(r"pid=(\d+)", href)
                if m_pid:
                    pid = m_pid.group(1)
                    break
            if not pid:
                pid = _make_pid(tid, page, floor, content)

            posted_at = ""
            row_text = row.get_text(" ", strip=True)
            dt = re.search(r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?", row_text)
            if dt:
                posted_at = dt.group(0)

            posts.append(
                NgaPost(
                    uid=uid,
                    tid=tid,
                    page=page,
                    floor=floor,
                    pid=pid,
                    author=author or "[-阿狼-]",
                    posted_at=posted_at,
                    content=content,
                    url=f"{base}/read.php?tid={tid}&authorid={uid}&page={page}",
                )
            )
        return posts

    def _fetch_page(self, uid: int, tid: int, page: int) -> list[NgaPost]:
        attempts: list[str] = []
        for base in self.base_urls:
            # First try NGA's UTF-8 JSON read API. It is less fragile than HTML and
            # works better from cloud runners on some NGA front doors.
            try:
                r = self._get(
                    base,
                    "/read.php",
                    {
                        "tid": tid,
                        "authorid": uid,
                        "page": page,
                        "__output": 11,
                        "noprefix": "",
                        "v2": "",
                    },
                )
                posts, blocked = self._parse_api_page(r.content, uid=uid, tid=tid, page=page, base=base)
                if posts:
                    self.base_url = base
                    return posts
                if not blocked:
                    # Valid API response with no author rows normally means end page.
                    text = _decode_bytes(r.content, "utf-8")
                    if '"__R"' in text or '"__PAGE"' in text:
                        self.base_url = base
                        return []
                attempts.append(f"{base}:api_blocked" if blocked else f"{base}:api_unparsed")
            except Exception as exc:
                attempts.append(f"{base}:api_{type(exc).__name__}")

            # Fallback to the HTML author-only view, as NGA occasionally returns
            # incomplete JSON for read.php.
            try:
                r = self._get(
                    base,
                    "/read.php",
                    {"tid": tid, "authorid": uid, "page": page, "noBBCode": ""},
                )
                html = _decode_bytes(r.content, r.encoding)
                if self._looks_blocked(html):
                    attempts.append(f"{base}:html_blocked")
                    continue
                posts = self._parse_author_only_page(html, uid=uid, tid=tid, page=page, base=base)
                self.base_url = base
                return posts
            except Exception as exc:
                attempts.append(f"{base}:html_{type(exc).__name__}")

        auth_present = bool(self.cookie_map.get("ngaPassportUid") and self.cookie_map.get("ngaPassportCid"))
        raise RuntimeError(
            "NGA authentication failed from all cloud endpoints. "
            f"auth_cookie_fields_present={auth_present}; attempts={','.join(attempts)}. "
            "Refresh ngaPassportUid/ngaPassportCid from a logged-in browser if needed."
        )

    def crawl_thread_author(
        self,
        uid: int = DEFAULT_UID,
        tid: int = DEFAULT_TID,
        start_page: int = 1,
        max_pages: int = 5000,
    ) -> Iterable[NgaPost]:
        """Crawl one author's replies in one thread with cloud-friendly endpoint fallback."""
        last_signature: tuple[str, ...] | None = None
        seen_pid: set[str] = set()

        for page in range(start_page, start_page + max_pages):
            posts = self._fetch_page(uid=uid, tid=tid, page=page)
            signature = tuple(p.pid for p in posts)
            if not posts or signature == last_signature:
                break
            last_signature = signature

            new_count = 0
            for post in posts:
                if post.pid in seen_pid:
                    continue
                seen_pid.add(post.pid)
                new_count += 1
                yield post

            if new_count == 0:
                break
            time.sleep(self.sleep_seconds)


def append_jsonl(path: Path, posts: Iterable[NgaPost]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(str(json.loads(line).get("pid", "")))
            except Exception:
                pass

    n = 0
    with path.open("a", encoding="utf-8") as f:
        for post in posts:
            if post.pid in existing:
                continue
            f.write(json.dumps(asdict(post), ensure_ascii=False) + "\n")
            existing.add(post.pid)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl NGA posts by author")
    parser.add_argument("--uid", type=int, default=DEFAULT_UID)
    parser.add_argument("--tid", type=int, default=DEFAULT_TID)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--base-url", default=os.getenv("NGA_BASE_URL", DEFAULT_BASE))
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "data" / "nga_posts.jsonl"),
    )
    args = parser.parse_args()

    crawler = NgaCrawler(base_url=args.base_url, sleep_seconds=args.sleep)
    out = Path(args.out)
    n = append_jsonl(
        out,
        crawler.crawl_thread_author(
            uid=args.uid,
            tid=args.tid,
            start_page=args.start_page,
            max_pages=args.max_pages,
        ),
    )
    print(f"added={n} out={out} endpoint={crawler.base_url}")


if __name__ == "__main__":
    main()
