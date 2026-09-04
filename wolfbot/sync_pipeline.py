from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jieba

from crawler import DEFAULT_TID, DEFAULT_UID, NgaCrawler, NgaPost

ROOT = Path(__file__).resolve().parent
KNOWLEDGE = ROOT / "knowledge"
STATE_PATH = KNOWLEDGE / "state.json"
RULES_PATH = KNOWLEDGE / "current_rules.json"
SOURCE_INDEX_PATH = KNOWLEDGE / "source_index.jsonl"
PENDING_RULES_PATH = KNOWLEDGE / "pending_rules.jsonl"
CONFLICTS_PATH = KNOWLEDGE / "conflicts.jsonl"

DEFAULT_CUTOFF = "2026-06-26"


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "trend": ("趋势", "反抽", "反弹", "企稳", "止跌", "颈线", "新高", "破位", "突破", "下跌浪"),
    "volume_price": ("缩量", "放量", "量能", "量价", "成交量", "补量"),
    "position": ("仓位", "加仓", "减仓", "满仓", "底仓", "清仓"),
    "t_trade": ("做T", "做 t", "日内", "滚动", "T出", "T 出"),
    "risk": ("止损", "风险", "暴雷", "减持", "ST", "退市", "保命", "回撤"),
    "market": ("大盘", "指数", "黄线", "白线", "GJD", "两融", "ETF", "A50", "期货"),
    "sector": ("板块", "主线", "轮动", "梯队", "补涨", "科技", "半导体", "CPO", "PCB"),
    "selection": ("选股", "核心", "强势票", "弱势票", "二线", "相对确定"),
    "events": ("消息", "会议", "数据", "业绩", "财报", "投资日历", "政策", "监管"),
    "psychology": ("心态", "恐慌", "贪婪", "情绪", "知行合一"),
    "special_regime": ("特殊时期", "量化", "战争", "黑天鹅", "特殊应对"),
}

RULE_HINTS = (
    "应该", "必须", "不要", "不能", "只要", "条件", "策略", "买点", "卖点", "止损",
    "加仓", "减仓", "仓位", "趋势", "缩量", "放量", "突破", "反抽", "反弹", "做T",
    "高开", "低开", "黄线", "白线", "资金", "主线", "板块", "记住", "我说过",
)

CONFLICT_DECISIONS = {
    "parameter_update",
    "scope_narrowing",
    "scope_expansion",
    "temporary_override",
    "supersede",
    "unresolved",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def append_jsonl_unique(path: Path, rows: list[dict[str, Any]], key: str) -> int:
    current = read_jsonl(path)
    seen = {str(x.get(key, "")) for x in current}
    added = 0
    for row in rows:
        k = str(row.get(key, ""))
        if not k or k in seen:
            continue
        current.append(row)
        seen.add(k)
        added += 1
    if added:
        write_jsonl(path, current)
    return added


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def short_excerpt(text: str, max_tokens: int = 20) -> str:
    """Keep only a very short exact excerpt; full forum text is not committed."""
    tokens = [x for x in jieba.lcut(text) if x.strip()]
    return "".join(tokens[:max_tokens]).strip()


def tags_for(text: str) -> list[str]:
    tags: list[str] = []
    low = text.lower()
    for tag, words in TOPIC_KEYWORDS.items():
        if any(w.lower() in low for w in words):
            tags.append(tag)
    return tags


def looks_like_rule(text: str) -> bool:
    return any(x in text for x in RULE_HINTS)


def parse_post_date(posted_at: str) -> str | None:
    if not posted_at:
        return None
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", posted_at)
    return m.group(1) if m else None


def current_rule_context() -> list[dict[str, str]]:
    obj = load_json(RULES_PATH, {})
    rows = []
    for r in obj.get("rules", []):
        if r.get("status") != "active":
            continue
        rows.append(
            {
                "id": str(r.get("id", "")),
                "category": str(r.get("category", "")),
                "title": str(r.get("title", "")),
                "statement": str(r.get("statement", "")),
                "scope": str(r.get("scope", "")),
            }
        )
    return rows


def build_llm_client():
    key = os.getenv("RULE_LLM_API_KEY", "").strip()
    model = os.getenv("RULE_LLM_MODEL", "").strip()
    if not key or not model:
        return None, None
    try:
        from openai import OpenAI
    except Exception:
        return None, None
    base_url = os.getenv("RULE_LLM_BASE_URL", "").strip()
    kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), model


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}


def llm_extract(post: NgaPost, rules: list[dict[str, str]]) -> dict[str, Any] | None:
    client, model = build_llm_client()
    if client is None or model is None:
        return None

    system = """你是交易体系版本管理员。输入是一条NGA用户阿狼的原始发言和当前规则库。
任务不是模仿语气，而是判断这条发言是否形成可复用交易规则，并判断与旧规则的关系。
严格区分：通用规则、特定市场环境的临时应对、参数更新、单次个股/盘中案例、纯情绪表达。
不能把案例擅自抽象成规则；不能因为表面措辞不同就制造冲突。
新旧说法冲突时先检查scope：长线/短线、主升/调整、机构行情/情绪行情、正常市场/特殊量化或政策环境。
如果明确说旧方法错了/要纠偏/修正，才允许supersede；如果说特殊时期特殊应对，应判temporary_override。
只输出JSON，不引用长段原文，不添加输入中不存在的事实。"""

    payload = {
        "current_rules": rules,
        "new_post": {
            "pid": post.pid,
            "posted_at": post.posted_at,
            "page": post.page,
            "content": post.content,
        },
        "output_schema": {
            "is_rule": "boolean",
            "category": "strategy|trend|volume_price|position|t_trade|risk|market|sector|selection|events|psychology|special_regime|other",
            "rule_summary": "中文短句，必须是转述，不是长引用",
            "scope": "适用市场环境/周期；无法确认填unknown",
            "conditions": ["触发条件"],
            "action": "对应动作；无则空字符串",
            "invalidation": ["失效/反向条件"],
            "decision": "reinforce|new_rule|parameter_update|scope_narrowing|scope_expansion|temporary_override|supersede|example_only|no_rule|unresolved",
            "related_rule_ids": ["Rxxx"],
            "conflict_reason": "无冲突则空字符串",
            "explicit_self_correction": "boolean",
            "time_specific": "boolean",
            "confidence": "0到1之间数字"
        }
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    try:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception:
            resp = client.chat.completions.create(model=model, temperature=0, messages=messages)
        text = resp.choices[0].message.content or ""
        obj = parse_json_object(text)
        if not obj:
            return None
        confidence = obj.get("confidence", 0)
        try:
            obj["confidence"] = max(0.0, min(1.0, float(confidence)))
        except Exception:
            obj["confidence"] = 0.0
        return obj
    except Exception as exc:
        return {"error": str(exc), "is_rule": False, "decision": "unresolved", "confidence": 0.0}


def source_record(post: NgaPost, extraction: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "source_id": f"nga:{post.tid}:{post.pid}",
        "uid": post.uid,
        "tid": post.tid,
        "pid": post.pid,
        "page": post.page,
        "floor": post.floor,
        "author": post.author,
        "posted_at": post.posted_at,
        "url": post.url,
        "content_sha256": content_hash(post.content),
        "excerpt": short_excerpt(post.content),
        "tags": tags_for(post.content),
        "rule_like_heuristic": looks_like_rule(post.content),
        "extraction_status": "done" if extraction is not None and "error" not in extraction else "pending",
        "extraction": extraction,
        "indexed_at": now_iso(),
    }


def candidate_record(src: dict[str, Any]) -> dict[str, Any] | None:
    ext = src.get("extraction") or {}
    if not ext:
        if not src.get("rule_like_heuristic"):
            return None
        return {
            "candidate_id": f"cand:{src['pid']}",
            "source_id": src["source_id"],
            "source_pid": src["pid"],
            "posted_at": src.get("posted_at", ""),
            "url": src.get("url", ""),
            "tags": src.get("tags", []),
            "status": "needs_project_review",
            "reason": "未配置RULE_LLM或抽取失败；仅按关键词判定为疑似规则。",
            "created_at": now_iso(),
        }
    if not bool(ext.get("is_rule")):
        return None
    return {
        "candidate_id": f"cand:{src['pid']}",
        "source_id": src["source_id"],
        "source_pid": src["pid"],
        "posted_at": src.get("posted_at", ""),
        "url": src.get("url", ""),
        "tags": src.get("tags", []),
        "status": "needs_project_review",
        "proposal": ext,
        "created_at": now_iso(),
    }


def main() -> None:
    KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    state = load_json(
        STATE_PATH,
        {
            "schema_version": 1,
            "baseline_cutoff": DEFAULT_CUTOFF,
            "last_author_page": 1,
        },
    )
    cutoff = str(state.get("baseline_cutoff") or DEFAULT_CUTOFF)
    existing_sources = read_jsonl(SOURCE_INDEX_PATH)
    seen = {str(x.get("pid", "")) for x in existing_sources}

    is_bootstrap = not existing_sources
    if is_bootstrap:
        start_page = 1
        max_pages = int(os.getenv("NGA_BOOTSTRAP_MAX_PAGES", "5000"))
    else:
        start_page = max(1, int(state.get("last_author_page", 1)) - 2)
        max_pages = int(os.getenv("NGA_SYNC_MAX_PAGES", "12"))

    crawler = NgaCrawler(sleep_seconds=float(os.getenv("NGA_SYNC_SLEEP", "1.0")))
    crawled = list(
        crawler.crawl_thread_author(
            uid=int(os.getenv("NGA_UID", str(DEFAULT_UID))),
            tid=int(os.getenv("NGA_TID", str(DEFAULT_TID))),
            start_page=start_page,
            max_pages=max_pages,
        )
    )

    max_page = max([p.page for p in crawled], default=int(state.get("last_author_page", 1)))
    rules = current_rule_context()
    new_sources: list[dict[str, Any]] = []
    new_candidates: list[dict[str, Any]] = []
    new_conflicts: list[dict[str, Any]] = []

    for post in crawled:
        if post.pid in seen:
            continue
        post_date = parse_post_date(post.posted_at)
        if post_date and post_date <= cutoff:
            continue
        # If the date is unavailable during bootstrap, only keep the tail pages to avoid
        # re-indexing the entire pre-PDF history as "new".
        if is_bootstrap and not post_date and post.page < max(1, max_page - 20):
            continue

        extraction = llm_extract(post, rules)
        src = source_record(post, extraction)
        new_sources.append(src)
        seen.add(post.pid)

        cand = candidate_record(src)
        if cand:
            new_candidates.append(cand)
            proposal = cand.get("proposal") or {}
            if proposal.get("decision") in CONFLICT_DECISIONS:
                new_conflicts.append(
                    {
                        "conflict_id": f"conflict:{post.pid}",
                        "candidate_id": cand["candidate_id"],
                        "source_pid": post.pid,
                        "decision": proposal.get("decision"),
                        "related_rule_ids": proposal.get("related_rule_ids", []),
                        "reason": proposal.get("conflict_reason", ""),
                        "status": "needs_project_review",
                        "created_at": now_iso(),
                    }
                )

    added_sources = append_jsonl_unique(SOURCE_INDEX_PATH, new_sources, "source_id")
    added_candidates = append_jsonl_unique(PENDING_RULES_PATH, new_candidates, "candidate_id")
    added_conflicts = append_jsonl_unique(CONFLICTS_PATH, new_conflicts, "conflict_id")

    state.update(
        {
            "last_run_at": now_iso(),
            "last_author_page": max(int(state.get("last_author_page", 1)), max_page),
            "last_seen_pid": new_sources[-1]["pid"] if new_sources else state.get("last_seen_pid"),
            "new_posts_last_run": added_sources,
            "new_rule_candidates_last_run": added_candidates,
            "new_conflicts_last_run": added_conflicts,
        }
    )
    if crawled:
        state["last_success_at"] = now_iso()
    save_json(STATE_PATH, state)

    print(
        json.dumps(
            {
                "bootstrap": is_bootstrap,
                "start_page": start_page,
                "max_page_seen": max_page,
                "new_sources": added_sources,
                "new_candidates": added_candidates,
                "new_conflicts": added_conflicts,
                "llm_enabled": build_llm_client()[0] is not None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
