# stock_monitor

A 股盘中“开盘状态 + 当天高点/低点”机械回放与实时信号研究仓库。

## Canonical repository

从 2026-09-03 起，本项目唯一后续更新位置为：

- Repository: `GeorgeCao1987/stock_monitor`
- Branch: `main`

旧仓库 `GeorgeCao1987/microservice` 的 `market-data-v13` 分支仅作为历史来源，不再继续更新本模型。

## 对话/开发恢复顺序

任何新对话、换模型或换执行环境后，继续本项目之前必须：

1. 先读取根目录 `STRATEGY.md`；
2. 查看 `main` 最新 commit；
3. 查看 `market_data/` 当前最高版本脚本；
4. 检查 `external_event_context.py`、`news_collector.py` 与外围实时消息覆盖；
5. 查看最近 GitHub Actions 回测结果；
6. 确认开发集、验证集和 Holdout 是否已被使用，再决定是否允许继续调参。

## Repository layout

- `STRATEGY.md`：长期策略上下文、不可违反的回测原则、已验证结论、版本迭代和待办。
- `market_data/`：数据采集、机械候选生成、V1.3～V1.9 回放与诊断代码。
- `market_data/external_event_context.py`：外围实时消息的决策时点过滤、结构化和上下文挂接。
- `market_data/news_collector.py`：外围宏观、AI/半导体、日韩、地缘等实时/历史消息采集与初步结构化。
- `.github/workflows/`：当前仍需要直接运行的 V1.8/V1.9 与韩国市场数据探测任务，默认手工触发。
- `workflow_archive/`：从旧仓库完整迁移的历史 GitHub Actions workflow，保留原始内容用于追溯。

## Current direction

当前主要方向以 `STRATEGY.md` 为准：

- 开盘层：判断 10:00 后哪一侧日内极值更可能尚未完成；
- HIGH：以 `WATCH_START` 候选预警为主，后续结构确认负责验证；
- LOW：以 `STRUCTURE_CONFIRM` 为主要确认阶段；
- 外围信息拆成两层：**外围价格先验** 与 **外围实时消息/事件先验**，不得混用；
- 10:00 判断只能使用 10:00 前已公开消息，盘中候选只能使用候选时间戳前已公开消息；
- 消息层当前作为正式条件进入采集与分层回放，但模型权重仍为 0，必须先做 BASELINE vs EVENT 增量回测；
- 所有回测必须每日机械产生全部候选，禁止挑事件、禁止未来函数；
- V1.9 下一步优先完成全部机械候选汇总、消息增量回测和新的独立 Holdout；
- 2025 年 11–12 月已被使用，不能再作为新消息层的 untouched holdout。
