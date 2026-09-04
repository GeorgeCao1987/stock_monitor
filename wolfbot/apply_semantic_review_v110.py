from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
K = ROOT / "knowledge"
RULES = K / "current_rules.json"
CONFLICTS = K / "conflicts.jsonl"
CHANGELOG = K / "CHANGELOG.md"
VERSIONS = K / "versions"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_evidence(rule: dict, ev: dict):
    arr = rule.setdefault("evidence", [])
    key = (ev.get("source_id"), ev.get("date"), ev.get("note"))
    if not any((x.get("source_id"), x.get("date"), x.get("note")) == key for x in arr if isinstance(x, dict)):
        arr.append(ev)


def main():
    doc = load_json(RULES)
    by_id = {x.get("id"): x for x in doc.get("rules", []) if isinstance(x, dict)}

    # Existing rules: evidence/clarification only; do not erase PDF evidence.
    r8 = by_id.get("R008")
    if r8:
        add_evidence(r8, {
            "source_id": "nga:47288722:880636850:2130baa9acbcaa39ef1401e1f421348ca8bacd6c3b36a6824b1bf9ad0e5cf780",
            "source": "NGA只看楼主 tid=47288722",
            "date": "2026-09-04 13:15",
            "note": "再次警示连续补仓后在低位缩量割肉、再追入另一风格方向的非理性仓位迁移。"
        })

    r13 = by_id.get("R013")
    if r13:
        r13["statement"] = "先保证大盘环境、板块阶段和底仓逻辑正确，再考虑T；T可以是日内也可以是隔1-3日的小波段差价，不要求卖出后马上买回；频繁T不能破坏最初买入与最终卖出逻辑。"
        r13["scope"] = "t_trade_intraday_or_short_swing"
        add_evidence(r13, {
            "source_id": "nga:47288722:880642206:7d52d7df956d1eb224581778d01e79859d571b0161ebc39730cfaab4c16a787b",
            "source": "NGA只看楼主 tid=47288722",
            "date": "2026-09-04 14:05",
            "note": "明确T不必限定当天，可以隔3日；反对卖出后立即机械买回。"
        })
        add_evidence(r13, {
            "source_id": "nga:47288722:880647858:ded353b0ab06cb16b690843661d88165a48d48fa10dd70c46a6a3f2a9d63da9f",
            "source": "NGA只看楼主 tid=47288722",
            "date": "2026-09-04 14:50",
            "note": "做T有输有赢，真正决定利润能否带走的是板块小波段的一买和一卖。"
        })

    new_rules = [
        {
            "id": "R018",
            "category": "regime_selection",
            "title": "低位看逻辑，高位看量价",
            "statement": "低位阶段可以用产业/基本面逻辑帮助筛选和建立预期；进入高位后，持有与退出必须更服从技术结构、筹码和量价，不能再用基本面逻辑替代卖出纪律。",
            "scope": "position_stage",
            "priority": "core",
            "status": "active",
            "decision": "new_rule",
            "evidence": [
                {
                    "source_id": "nga:47288722:0:c37ed43789fb4795efdb6bc654dd4f2fb532cb9e471a40f2e4f711c5fdfd226d",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-07-31 15:10",
                    "note": "新帖开篇明确总结：操作跟策略、策略跟市场；低位看逻辑，高位看量价。"
                },
                {
                    "source_id": "nga:47288722:877189699:4620b2bda56bf7ca9c45984cda817b4645420c5b4474ee9b978565ffef2367c9",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-08-02 17:29",
                    "note": "再次强调高位不再看基本面逻辑，转而看纯技术图形与量价关系。"
                }
            ]
        },
        {
            "id": "R019",
            "category": "timeframe_regime",
            "title": "大级别定交易模式，小级别服从大级别",
            "statement": "先定义指数/市场所处的大级别阶段，再决定能否预期主升、反转或仅做反抽；如果大级别被定义为调整浪反弹，小级别板块和个股不能仅因短时走强就直接按主升模式操作。",
            "scope": "multi_timeframe_regime",
            "priority": "core",
            "status": "active",
            "decision": "new_rule",
            "evidence": [
                {
                    "source_id": "nga:47288722:880010157:5f76d1245e8a4f876e7b090ab882cb6b992a1a505df5976550338ed85dea6534",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-08-29 16:09",
                    "note": "明确提出小级别服从大级别；4-4被定义为反弹区间时，操作基调是吃反弹而不是预期板块主升。"
                }
            ]
        },
        {
            "id": "R020",
            "category": "regime_position",
            "title": "调整浪反抽中的高低位分层处理",
            "statement": "在明确属于调整浪反抽、而非主升的场景中，高位旧方向以做T、降低成本和反抽退出为主，低位方向可以持仓等待或寻找辨识度；不能把高位反抽按主升越涨越加仓。",
            "scope": "adjustment_wave_rebound",
            "priority": "high",
            "status": "active",
            "decision": "scope_narrowing",
            "related_rule_ids": ["R007", "R011", "R013", "R019"],
            "evidence": [
                {
                    "source_id": "nga:47288722:877366337:f032b59f9da322203a031eb5560f860f3a1996a25bd8d0c0e03eefa47a8233bb",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-08-04 12:15",
                    "note": "明确给出调整阶段的基础逻辑：高位方向做T、低位持仓等，方向可自选但持仓占比要调整。"
                },
                {
                    "source_id": "nga:47288722:880641092:8961b25b84ee674975ddd6ba8caced8085e20cc7524324e00c7e219c7921660d",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-09-04 13:55",
                    "note": "再次强调这是调整浪反抽，大资金按做T降低成本，不能按主升不断拉升的模式理解。"
                },
                {
                    "source_id": "nga:47288722:880650211:98f5f215be5abfba1efec3dc1d9126efcec5ce2ea1ca79e8099eed569a519db8",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-09-04 15:07",
                    "note": "阶段总结进一步区分此前高位大科技与低位AI软/券商/军工的不同处理方式。"
                }
            ]
        },
        {
            "id": "R021",
            "category": "special_pattern",
            "title": "双跌停战法必须机械满足预设条件",
            "statement": "双跌停属于高风险、纯资金博弈的特殊模式；只在预设条件全部满足时机械执行，不满足就不做，不能临场加入基本面想象改变条件。",
            "scope": "double_limit_down_pattern_only",
            "priority": "medium",
            "status": "active",
            "decision": "reinforce_and_formalize",
            "risk_level": "high",
            "evidence": [
                {
                    "source": "自立自强.pdf",
                    "date": "2026-01-19/02-09",
                    "section": "双跌停战法",
                    "note": "原PDF已说明该方法风险高、不是常用方法，且要理解形成K线组合的资金逻辑。"
                },
                {
                    "source_id": "nga:47288722:880642743:d3418007f7a47f9307b8e09cdf7f4e7205804ad1de129b12ad7e1b71f6baf96b",
                    "source": "NGA只看楼主 tid=47288722",
                    "date": "2026-09-04 14:09",
                    "note": "再次明确双跌停是赌模式内成功率，预设条件全是纯资金；达成就做，不达成就不做。"
                }
            ]
        }
    ]

    for nr in new_rules:
        if nr["id"] not in by_id:
            doc.setdefault("rules", []).append(nr)
            by_id[nr["id"]] = nr

    doc["version"] = "1.1.0"
    doc["version_date"] = "2026-09-04"
    doc["latest_verified_source_at"] = "2026-09-04T15:09:00+08:00"
    doc["latest_verified_source_id"] = "nga:47288722:880650450:e48d488913db3de006489e3e2f12e898f6a95336e84c5ecd8e83c88dc4ea0bf2"
    doc["source_policy"] = "PDF为不可变基线；NGA tid=47288722 authorid=150058 的原始发言按每帖独立不可变快照归档；派生规则经语义审阅后进入active。"
    doc["version_change"] = {
        "type": "minor",
        "summary": "完整归档当前只看楼主线程784条发言（2026-07-31至2026-09-04），新增低位看逻辑/高位看量价、大级别定交易模式、调整浪高低位分层等规则，并扩展做T为日内或短波段。"
    }
    RULES.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    conflict_id = "conf:R011:R020:20260904"
    existing = CONFLICTS.read_text(encoding="utf-8") if CONFLICTS.exists() else ""
    if conflict_id not in existing:
        rec = {
            "conflict_id": conflict_id,
            "status": "resolved_by_scope",
            "annotation_at": "2026-09-04T20:00:00+08:00",
            "older_rule_id": "R011",
            "new_rule_id": "R020",
            "decision": "scope_narrowing",
            "explanation": "PDF的一般反弹规则是卖弱留强；2026-09-04对此前高位大科技的阶段总结出现卖强留弱、拉升后都走。该说法处于调整浪反抽/高位旧方向退出语境，只覆盖该scope，不推翻一般反弹规则。",
            "source_ids": [
                "nga:47288722:880650211:98f5f215be5abfba1efec3dc1d9126efcec5ce2ea1ca79e8099eed569a519db8"
            ]
        }
        with CONFLICTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    VERSIONS.mkdir(parents=True, exist_ok=True)
    (VERSIONS / "1.1.0.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cl = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else "# 阿狼体系版本记录\n"
    marker = "## 1.1.0 — 2026-09-04"
    if marker not in cl:
        entry = '''\n## 1.1.0 — 2026-09-04\n\n切换到用户提供的当前只看楼主线程 `tid=47288722&authorid=150058`，完整归档40页、784条阿狼原始发言，时间覆盖2026-07-31 15:10至2026-09-04 15:09。原始发言逐帖独立保存并记录SHA-256，不覆盖、不改写。\n\n### New / clarified\n- 新增 R018：低位看逻辑，高位看量价。\n- 新增 R019：大级别定交易模式，小级别服从大级别。\n- 新增 R020：调整浪反抽中的高低位分层处理。\n- 新增 R021：双跌停战法必须机械满足预设条件，且仅限高风险特殊模式。\n- 扩展 R013：做T不限定日内，可为隔1-3日短波段；不要求卖出后立即买回。\n- R011 与 R020 的表面冲突按 scope_narrowing 处理：一般反弹仍卖弱留强；高位旧方向在调整浪反抽退出阶段可优先兑现强势反抽，最终拉升后退出。\n\n'''
        if cl.startswith("# 阿狼体系版本记录"):
            first, rest = cl.split("\n", 1)
            cl = first + "\n" + entry + rest
        else:
            cl = entry + cl
        CHANGELOG.write_text(cl, encoding="utf-8")

    print(json.dumps({"version": doc["version"], "rules": len(doc.get("rules", [])), "conflict": conflict_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
