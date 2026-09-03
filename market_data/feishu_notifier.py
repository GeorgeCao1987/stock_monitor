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
    return f"【盘中监控】\n时间：{now}\n状态：连接正常"


def build_signal_card(data: dict) -> dict:
    title = str(data.get("卡片标题", "盘中监控"))
    header_color = str(data.get("标题颜色", "blue"))

    def value(key: str, default: str = "—") -> str:
        v = data.get(key, default)
        return default if v is None or v == "" else str(v)

    # 默认采用极简卡片：时间、现价、顶部、底部、结论。
    compact_keys = ["时间", "现价", "顶部", "底部"]
    if any(k in data for k in compact_keys):
        fields = [
            ("时间", value("时间")),
            ("现价", value("现价")),
            ("顶部", value("顶部")),
            ("底部", value("底部")),
        ]
        field_nodes = [
            {
                "is_short": True,
                "text": {"tag": "lark_md", "content": f"**{name}**\n{val}"},
            }
            for name, val in fields
        ]
        elements = [
            {"tag": "div", "fields": field_nodes},
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**结论**  {value('结论')}"},
            },
        ]
        return {
            "config": {"wide_screen_mode": True, "enable_forward": True},
            "header": {
                "template": header_color,
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": elements,
        }

    # 兼容旧的详细卡片数据，但后续实盘默认不使用。
    fields = [
        ("证券名称", value("证券名称")),
        ("当前价格", value("当前价格")),
        ("顶部锁定概率", value("顶部锁定概率")),
        ("底部锁定概率", value("底部锁定概率")),
        ("当前状态", value("当前状态")),
    ]
    field_nodes = [
        {
            "is_short": True,
            "text": {"tag": "lark_md", "content": f"**{name}**\n{val}"},
        }
        for name, val in fields
    ]
    elements = [
        {"tag": "div", "fields": field_nodes},
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**结论**  {value('操作结论')}"},
        },
    ]
    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": header_color,
            "title": {"tag": "plain_text", "content": title},
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
