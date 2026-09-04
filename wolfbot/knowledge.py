from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jieba
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


@dataclass
class Doc:
    doc_id: str
    source: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\xa0", " ")
    text = re.sub(r"[\t\r ]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    text = clean(text)
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return out


def load_pdf(path: Path) -> list[Doc]:
    if not path.exists():
        return []
    docs: list[Doc] = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
        for idx, chunk in enumerate(split_text(page.extract_text() or "")):
            docs.append(Doc(f"pdf-{page_no}-{idx}", "pdf", f"自立自强 P{page_no}", chunk, {"page": page_no}))
    return docs


def load_nga(path: Path) -> list[Doc]:
    if not path.exists():
        return []
    docs: list[Doc] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except Exception:
            continue
        text = clean(str(row.get("content") or ""))
        if text:
            docs.append(Doc(
                f"nga-{row.get('pid') or line_no}", "nga",
                f"NGA 阿狼 {row.get('posted_at') or ''}".strip(), text,
                {k: row.get(k) for k in ("uid", "tid", "page", "floor", "pid", "url")},
            ))
    return docs


def tokens(text: str) -> list[str]:
    text = text.lower()
    out = [x.strip() for x in jieba.lcut(text) if x.strip()]
    out.extend(re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?", text, flags=re.I))
    return out


class KnowledgeBase:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokens(d.title + " " + d.text) for d in docs]) if docs else None

    @classmethod
    def build(cls, pdf_path: Path | None, nga_path: Path | None):
        docs: list[Doc] = []
        if pdf_path:
            docs += load_pdf(pdf_path)
        if nga_path:
            docs += load_nga(nga_path)
        return cls(docs)

    def retrieve(self, query: str, k: int = 8) -> list[Doc]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokens(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in idx[:k] if scores[i] > 0]

    def stats(self) -> dict[str, int]:
        ans: dict[str, int] = {}
        for d in self.docs:
            ans[d.source] = ans.get(d.source, 0) + 1
        return ans
