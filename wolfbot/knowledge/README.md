# 阿狼体系动态知识库

这个目录不是“语录堆积”，而是把新增 NGA 发言转换成可追溯、可冲突处理、可版本化的交易体系。

## 数据流

1. **Source / 原始来源层**
   - NGA 目标主题：`tid=45974302`
   - 目标作者：`uid=150058`（阿狼）
   - PDF《自立自强》作为 V1.0 基线，当前基线截止到 2026-06-26 左右。
   - 云端采集不把完整论坛正文提交到公开仓库；仅保留来源指针、内容哈希、短摘录和结构化抽取结果。

2. **Pending / 候选规则层**
   - `source_index.jsonl`：每条新发言的来源索引。
   - `pending_rules.jsonl`：疑似规则或方法论的新发言。
   - `conflicts.jsonl`：与现有体系存在潜在冲突/覆盖关系的候选。

3. **Active / 当前体系层**
   - `current_rules.json`：机器人实际使用的当前规则版本。
   - 新发言不会直接覆盖旧规则；先按 `conflict_policy.json` 做语义审阅。

4. **Version / 版本层**
   - `versions/`：固化后的历史规则快照。
   - `CHANGELOG.md`：记录新增、修改、特例覆盖、废弃/被替代。

## 冲突处理原则

优先区分“真正改规则”和“换了市场环境”：

- 明确自我纠偏/修正：新规则可以 supersede 旧规则。
- 特殊时期特殊应对：记为 temporary override，不覆盖常规规则。
- 长线/短线、主升/调整、机构行情/情绪行情等：先做 scope 拆分，避免伪冲突。
- 点位、均线、仓位、量能阈值变化：默认视为 parameter update。
- 单次个股点评/盘中案例：默认 example_only，不自动升级为通用规则。
- 无法判断：保留双方并标记 unresolved，禁止模型擅自“统一口径”。

## 项目侧机器人应如何使用

ChatGPT 项目的项目指令里建议增加下面这段：

> 在回答任何“阿狼模式”的股票问题前，优先从 GitHub `GeorgeCao1987/stock_monitor` 的 `wolfbot` 分支读取 `wolfbot/knowledge/current_rules.json`。如果存在 `pending_rules.jsonl` 或 `conflicts.jsonl` 的未处理增量，先按照 `conflict_policy.json` 做语义审阅，再形成当前回答。任何新规则必须保留 source pid/url/日期；不能把模型推断冒充阿狼原话。若新发言只是特殊行情应对，只建立 conditional/temporary override，不覆盖 general 规则。

这样 ChatGPT Project 是前端和最终语义审阅器，GitHub 是云端版本库，NGA 采集由 GitHub Actions 运行；本地电脑不参与。

## GitHub Secrets

自动抓取至少需要：

- `NGA_COOKIE`：NGA 登录 Cookie（不要放进代码或聊天）。

若要后台直接做“规则提取 + 冲突候选判断”，再增加 OpenAI-compatible 模型配置：

- `RULE_LLM_API_KEY`
- `RULE_LLM_BASE_URL`（可选）
- `RULE_LLM_MODEL`

如果没有配置规则模型，采集仍会工作，但新增内容只进入“待项目侧审阅”状态。

## 版本号

- PATCH：文字澄清、证据增加、轻微参数更新。
- MINOR：新增规则、增加适用场景、增加临时覆盖。
- MAJOR：核心交易原则被阿狼明确推翻或重构。
