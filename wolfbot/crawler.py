from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE = "https://bbs.nga.cn"
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
    out: dict[str, str] = {}
    for part in cookie.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip()
    return out


class NgaCrawler:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        cookie: str | None = None,
        sleep_seconds: float = DEFAULT_SLEEP,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.getenv(
                    "NGA_USER_AGENT",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                ),
                "X-User-Agent": "Nga_Official",
                "Referer": self.base_url + "/",
            }
        )
        cookie = cookie or os.getenv("NGA_COOKIE", "")
        if cookie:
            self.session.cookies.update(_parse_cookie_string(cookie))

    def _get(self, path: str, params: dict) -> requests.Response:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r

    @staticmethod
    def _looks_blocked(html: str) -> bool:
        markers = ["版面关闭", "你可能需要", "登录后访问", "ERROR:2048"]
        return any(x in html for x in markers)

    def _parse_author_only_page(
        self,
        html: str,
        uid: int,
        tid: int,
        page: int,
    ) -> list[NgaPost]:
        soup = BeautifulSoup(html, "lxml")
        posts: list[NgaPost] = []

        rows = soup.select("[id^='postrow']")
        if not rows:
            # NGA occasionally changes wrappers; postcontent remains relatively stable.
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

            url = f"{self.base_url}/read.php?tid={tid}&authorid={uid}&page={page}"
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
                    url=url,
                )
            )
        return posts

    def crawl_thread_author(
        self,
        uid: int = DEFAULT_UID,
        tid: int = DEFAULT_TID,
        start_page: int = 1,
        max_pages: int = 5000,
    ) -> Iterable[NgaPost]:
        """Crawl only one author's replies in one thread.

        NGA supports read.php?tid=<tid>&authorid=<uid>&page=<n>, so we avoid
        traversing every ordinary thread page. Pagination stops when the next page
        contains no posts or repeats the previous page's post signature.
        """
        last_signature: tuple[str, ...] | None = None
        seen_pid: set[str] = set()

        for page in range(start_page, start_page + max_pages):
            r = self._get(
                "/read.php",
                {
                    "tid": tid,
                    "authorid": uid,
                    "page": page,
                    "noBBCode": "",
                },
            )
            html = r.text
            if self._looks_blocked(html):
                raise RuntimeError(
                    "NGA returned a login/permission page. Set NGA_COOKIE with your "
                    "browser cookies, e.g. ngaPassportUid=...; ngaPassportCid=..."
                )

            posts = self._parse_author_only_page(html, uid=uid, tid=tid, page=page)
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
    print(f"added={n} out={out}")


if __name__ == "__main__":
    main()
