import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


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
    return f"【Stock Monitor 推送测试】\n飞书机器人连接正常\n模型：V3.7\n类型：TEST\n时间：{now}\n状态：推送链路验证"


def send_text(webhook: str, message: str) -> dict:
    payload = {
        "msg_type": "text",
        "content": {"text": message},
    }
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


def main() -> int:
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook:
        print("FEISHU_WEBHOOK_URL secret is missing", file=sys.stderr)
        return 2

    message = load_message()
    if not message:
        print("message is empty", file=sys.stderr)
        return 3

    result = send_text(webhook, message)
    print("FEISHU_PUSH_OK", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
