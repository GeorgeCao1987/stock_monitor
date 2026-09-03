import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _read_json_file(path_value: str) -> dict:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError(f"card file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_message() -> str:
    message = os.getenv("FEISHU_MESSAGE", "").strip()
    message_file = os.getenv("FEISHU_MESSAGE_FILE", "").strip()
    if message:
        return message
    if message_file:
        path = Path(message_file)
        if not path.exists():
            raise RuntimeError(f"message file does not exist: {path}")
        return path.read_text(encoding="utf-8").strip()
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    return f"【Stock Monitor 推送测试】\n飞书机器人连接正常\n模型版本：V3.7\n消息类型：测试\n时间：{now}\n当前状态：推送链路验证"


def build_signal_card(data: dict) -> dict:
    title = str(data.get("卡片标题", "盘中T交易监控"))
    header_color = str(data.get("标题颜色", "blue"))

    def value(key: str, default: str = "—") -> str:
        v = data.get(key, default)
        return default if v is None or v == "" else str(v)

    fields = [
        ("证券名称", value("证券名称")),
        ("证券代码", value("证券代码")),
        ("模拟时间", value("模拟时间")),
        ("当前价格", value("当前价格")),
        ("顶部锁定概率", value("顶部锁定概率")),
        ("底部锁定概率", value("底部锁定概率")),
        ("距日内高点", value("距日内高点")),
        ("距日内低点", value("距日内低点")),
        ("反T交易价值", value("反T交易价值")),
        ("正T交易价值", value("正T交易价值")),
        ("当前状态", value("当前状态")),
        ("信号级别", value("信号级别")),
    ]

    field_nodes = [
        {
            "is_short": True,
            "text": {
                "tag": "lark_md",
                "content": f"**{name}**\n{val}",
            },
        }
        for name, val in fields
    ]

    elements = [
        {
            "tag": "div",
            "fields": field_nodes,
        },
        {"tag": "hr"},
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**操作结论**\n{value('操作结论')}",
            },
        },
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**关键证据**\n{value('关键证据')}",
            },
        },
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**失效条件**\n{value('失效条件')}",
            },
        },
        {"tag": "hr"},
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"{value('数据性质', '实盘')} ｜ 模型版本：{value('模型版本', 'V3.7')} ｜ 数据粒度：{value('数据粒度', '5分钟')}",
                }
            ],
        },
    ]

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": header_color,
            "title": {
                "tag": "plain_text",
                "content": title,
            },
        },
        "elements": elements,
    }


def load_card() -> dict | None:
    card_file = os.getenv("FEISHU_CARD_FILE", "").strip()
    card_json = os.getenv("FEISHU_CARD_JSON", "").strip()
    if card_json:
        return build_signal_card(json.loads(card_json))
    if card_file:
        return build_signal_card(_read_json_file(card_file))
    return None


def _post(webhook: str, payload: dict) -> dict:
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8", errors="replace")
            result = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu webhook HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Feishu webhook network error: {exc.reason}") from exc

    code = result.get("code", result.get("StatusCode", 0))
    if code not in (0, None):
        raise RuntimeError(f"Feishu webhook rejected message: {result}")
    return result


def send_text(webhook: str, message: str) -> dict:
    return _post(webhook, {"msg_type": "text", "content": {"text": message}})


def send_card(webhook: str, card: dict) -> dict:
    return _post(webhook, {"msg_type": "interactive", "card": card})


def main() -> int:
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook:
        print("FEISHU_WEBHOOK_URL secret is missing", file=sys.stderr)
        return 2

    card = load_card()
    if card is not None:
        result = send_card(webhook, card)
        print("FEISHU_CARD_PUSH_OK", json.dumps(result, ensure_ascii=False))
        return 0

    message = load_message()
    if not message:
        print("message is empty", file=sys.stderr)
        return 3

    result = send_text(webhook, message)
    print("FEISHU_PUSH_OK", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
