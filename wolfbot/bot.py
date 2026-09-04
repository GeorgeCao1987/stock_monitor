from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from knowledge import KnowledgeBase
from market import build_snapshot, normalize_code

load_dotenv()
BASE = Path(__file__).resolve().parent

SYSTEM_PROMPT = """你是“WolfBot / 阿狼交易体系研究助手”，不是阿狼本人，也不要冒充他。
你的任务是：基于提供的阿狼语料、结构化规则和当前行情数据，用这套思路分析股票和交易知识。

必须遵守：
1. 核心不是猜涨跌，而是给条件化策略：当出现A做B；不出现A就不做B；同时写失效条件。
2. 优先顺序：市场环境/趋势 -> 板块与资金 -> 个股 -> 量价与关键位 -> 仓位 -> 做T。不要跳过前面直接讨论日内技巧。
3. 日K判断趋势，15分钟/分时辅助日内。量价是技术判断基础。
4. 不能只看图。若缺少板块强弱、指数黄白线、消息、机构资金等关键数据，要明确写“缺数据”，不能编造。
5. 市场实际走势优先于预判。发现预判错误时，给纠偏条件。
6. 不追求最大利润，优先确定性与回撤控制。不要输出“必涨、稳赢、梭哈”等确定性措辞。
7. 引用语料时只引用上下文里真实存在的内容，不要制造阿狼原话。
8. 将内容区分为：【语料支持】【实时数据】【推断】。推断不能伪装成阿狼观点。

有股票代码时，默认输出：
- 狼式结论：一句话说明“现在更像什么阶段、应该等什么”。
- 1. 市场/趋势状态
- 2. 量价与关键位置
- 3. 板块/资金还需要验证什么
- 4. 条件化应对（触发条件 -> 动作；未触发 -> 动作）
- 5. 仓位思路（只给体系上的轻/中/重和前提，不代用户下单）
- 6. 失效条件与风险
- 7. 依据：列出本次检索到的语料来源标题/页码或NGA信息

如果用户问的是概念或方法，就围绕该问题解释，不强行套股票模板。
"""


def load_rules(path: Path | None = None) -> list[dict[str, str]]:
    path = path or (BASE / "rules.json")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "缺失"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def format_market(snapshot) -> str:
    if snapshot is None:
        return "无股票代码或行情获取失败。"
    d = snapshot.to_dict()
    return "\n".join([
        f"股票: {d['name']}({d['code']})",
        f"时间: {d['timestamp']} | 数据源: {d['source']}",
        f"现价: {_fmt(d['price'])} | 涨跌幅: {_fmt(d['pct'])}% | 昨收: {_fmt(d['prev_close'])}",
        f"今开/高/低: {_fmt(d['open'])} / {_fmt(d['high'])} / {_fmt(d['low'])}",
        f"MA5/10/13/20/60/144: {_fmt(d['ma5'])} / {_fmt(d['ma10'])} / {_fmt(d['ma13'])} / {_fmt(d['ma20'])} / {_fmt(d['ma60'])} / {_fmt(d['ma144'])}",
        f"近20日高/低: {_fmt(d['day20_high'])} / {_fmt(d['day20_low'])}",
        f"截至当前同时间成交额/上一交易日同期: {_fmt(d['amount_ratio_same_time'])}",
        f"机械趋势提示: {d['trend_hint']}",
    ])


def format_refs(refs) -> str:
    if not refs:
        return "未检索到语料。"
    blocks = []
    for i, d in enumerate(refs, 1):
        meta = ", ".join(f"{k}={v}" for k, v in d.meta.items() if v not in (None, ""))
        blocks.append(f"[{i}] {d.title} | {d.source} | {meta}\n{d.text}")
    return "\n\n".join(blocks)


class WolfBot:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.rules = load_rules()
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or None
        self.model = os.getenv("LLM_MODEL", "gpt-5.6")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

    def _market(self, question: str):
        code = normalize_code(question)
        if not code:
            return None
        try:
            return build_snapshot(code)
        except Exception:
            return None

    def _query(self, question: str, snapshot) -> str:
        extra = ""
        if snapshot:
            extra = f" {snapshot.name} {snapshot.code} {snapshot.trend_hint} 量价 趋势 仓位 板块 资金"
        return question + extra

    def _fallback(self, question: str, snapshot, refs) -> str:
        lines = ["## 狼式分析（本地规则模式）"]
        if snapshot:
            d = snapshot.to_dict()
            lines += [
                f"**{d['name']}({d['code']})**：机械趋势状态为 `{d['trend_hint']}`。",
                f"现价 {_fmt(d['price'])}；MA20 {_fmt(d['ma20'])}；MA60 {_fmt(d['ma60'])}；MA144 {_fmt(d['ma144'])}；同期成交额比 {_fmt(d['amount_ratio_same_time'])}。",
                "这只能解决个股量价的一部分。按阿狼体系，还必须验证指数、板块相对强弱、黄白线和资金/消息，缺这些数据时不把结论升级为明确交易信号。",
            ]
        else:
            lines.append("当前没有识别到股票代码，按知识问答处理。")

        lines.append("\n### 条件化思路")
        for r in self.rules[:8]:
            if any(x in question for x in r["name"]) or len(lines) < 8:
                lines.append(f"- **{r['name']}**：{r['rule']}")

        if refs:
            lines.append("\n### 本次检索到的语料")
            for d in refs[:5]:
                where = f"P{d.meta.get('page')}" if d.source == "pdf" else f"NGA page={d.meta.get('page')} floor={d.meta.get('floor')}"
                lines.append(f"- {d.title}（{where}）：{d.text[:180]}…")
        lines.append("\n> 当前未配置 LLM_API_KEY，因此这是规则+检索模式；配置兼容 OpenAI 的模型接口后会生成完整自然语言推理。")
        return "\n".join(lines)

    def answer(self, question: str) -> str:
        snapshot = self._market(question)
        refs = self.kb.retrieve(self._query(question, snapshot), k=8)
        if not self.client:
            return self._fallback(question, snapshot, refs)

        rules_text = "\n".join(f"- {r['name']}: {r['rule']}" for r in self.rules)
        user_prompt = f"""用户问题：
{question}

【实时/最近行情数据】
{format_market(snapshot)}

【结构化规则索引】
{rules_text}

【检索到的阿狼语料】
{format_refs(refs)}

请严格按系统要求回答。语料不足的地方标为“缺数据/推断”，不要补造阿狼观点。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


def build_default_bot() -> WolfBot:
    pdf_env = os.getenv("WOLFBOT_PDF", "").strip()
    pdf_path = Path(pdf_env) if pdf_env else (BASE / "data" / "自立自强.pdf")
    nga_path = Path(os.getenv("WOLFBOT_NGA", str(BASE / "data" / "nga_posts.jsonl")))
    kb = KnowledgeBase.build(pdf_path if pdf_path.exists() else None, nga_path if nga_path.exists() else None)
    return WolfBot(kb)
