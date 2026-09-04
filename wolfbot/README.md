# WolfBot：阿狼交易体系研究助手

这个目录把三类信息组合成一个小型股票分析机器人：

1. 《自立自强》PDF：作为已有历史语料；
2. NGA 阿狼发言：默认抓取 UID `150058` 在主题 `45974302` 中的“只看TA”内容；
3. 实时/最近行情：通过东方财富接口获取个股行情、日K、5分钟K，计算 MA5/10/13/20/60/144、近20日高低点和同期成交额比。

机器人不冒充阿狼。回答会尽量区分“语料支持、实时数据、推断”，核心输出采用条件策略，而不是直接猜涨跌。

## 1. 安装

```bash
git checkout wolfbot
cd wolfbot
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
```

## 2. 放入 PDF

创建 `data` 目录，并把附件放为：

```text
wolfbot/data/自立自强.pdf
```

也可以在 `.env` 里设置：

```text
WOLFBOT_PDF=D:/your/path/【狼大】自立自强.pdf
```

## 3. 爬取 NGA 阿狼发言

NGA 页面常要求登录。把你浏览器已经登录 NGA 后请求头里的 Cookie 复制到本机 `.env`：

```text
NGA_COOKIE=ngaPassportUid=...; ngaPassportCid=...; ...
```

Cookie 不要提交到 GitHub。

默认抓取：

```bash
python crawler.py
```

等价于：

```bash
python crawler.py --uid 150058 --tid 45974302
```

输出：

```text
data/nga_posts.jsonl
```

爬虫会：

- 使用 `read.php?tid=...&authorid=...&page=...` 只看该作者；
- 自动分页；
- 按 PID 去重，没有 PID 时生成内容哈希；
- 默认每页间隔 1.2 秒，避免高频访问；
- 再运行时只追加新内容。

建议先保留这个主题内的股票语料。相比全站抓 UID，它噪音更低，也更贴合《自立自强》的内容。

## 4. 配置模型

支持 OpenAI 兼容接口：

```text
LLM_API_KEY=你的key
LLM_BASE_URL=https://你的兼容接口/v1
LLM_MODEL=模型名
```

不配置模型也能运行，此时是“BM25 检索 + 结构化规则”的本地模式。

## 5. 启动机器人

```bash
streamlit run app.py
```

浏览器打开 Streamlit 地址后可以直接问：

```text
002916 深南电路现在按阿狼体系怎么看？
```

或者：

```text
缩量跌破为什么不能直接认为破位？
调整期仓位应该怎么处理？
反抽怎么确认升级成反弹？
今天适不适合做T，要看哪些条件？
```

## 6. 当前分析流程

```text
用户问题
  ↓
识别是否含6位股票代码
  ↓
东方财富行情：现价 / K线 / 均线 / 同期量能
  ↓
BM25检索：《自立自强》 + 最新NGA发言
  ↓
加载 rules.json 结构化体系
  ↓
LLM 生成条件化分析
  ↓
趋势 → 板块/资金 → 个股量价 → 仓位 → 做T → 失效条件
```

其中结构化规则只作为检索和提示索引，真正回答仍应优先回到原始语料，不把二次总结伪装成阿狼原话。

## 7. 目前刻意没有自动化的部分

阿狼体系多次强调不能只看个股K线，因此以下数据目前没有假装“自动知道”：

- 大盘黄线/白线关系；
- 当日主线板块和板块梯队；
- 机构/量化/GJD 的实时资金行为；
- 盘中突发消息及其影响方向；
- 龙虎榜和更细盘口语言。

Bot 在缺这些数据时应该明确写“缺数据”，而不是编造结论。后续可以把 `stock_monitor` 现有盘中模型、新闻采集和板块实时数据接进来，形成第二阶段。

## 8. 文件说明

- `crawler.py`：NGA 阿狼发言增量采集；
- `market.py`：东方财富行情与基础技术数据；
- `knowledge.py`：PDF/NGA 解析与 BM25 检索；
- `rules.json`：从语料中抽出的结构化规则索引；
- `bot.py`：机器人推理与模型调用；
- `app.py`：Streamlit 对话界面；
- `.env.example`：配置样例。
