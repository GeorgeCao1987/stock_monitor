import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from pytdx.hq import TdxHq_API

from oos_core_collect import server_list

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SYMBOL_NAMES = {
    "002916": "深南电路",
    "002463": "沪电股份",
    "600183": "生益科技",
    "002938": "鹏鼎控股",
    "603228": "景旺电子",
    "300476": "胜宏科技",
}


def fetch_5m(symbol: str, target_date: str) -> pd.DataFrame:
    market = 1 if symbol.startswith("6") else 0
    target = pd.Timestamp(target_date)
    for ip, port in server_list():
        api = None
        try:
            api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
            if not api.connect(ip, port, time_out=2):
                continue
            parts = []
            for page in range(6):
                rows = api.get_security_bars(0, market, symbol, page * 800, 800) or []
                if not rows:
                    break
                x = api.to_df(rows)
                x["ts"] = pd.to_datetime(x["datetime"], errors="coerce")
                parts.append(x)
                if x["ts"].min().normalize() <= target.normalize():
                    break
            if not parts:
                continue
            z = pd.concat(parts, ignore_index=True)
            z["ts"] = pd.to_datetime(z["datetime"], errors="coerce")
            z = z.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts")
            z = z[z["ts"].dt.strftime("%Y-%m-%d") == target_date].copy()
            if not z.empty:
                for c in ["open", "high", "low", "close", "vol", "amount"]:
                    if c in z.columns:
                        z[c] = pd.to_numeric(z[c], errors="coerce")
                return z
        except Exception as exc:
            print("SERVER_FAIL", ip, port, repr(exc))
        finally:
            if api is not None:
                try:
                    api.disconnect()
                except Exception:
                    pass
    raise RuntimeError(f"无法取得 {symbol} {target_date} 的5分钟行情")


def main():
    symbol = os.getenv("ASOF_SYMBOL", "002916").strip()
    date_s = os.getenv("ASOF_DATE", "2026-09-03").strip()
    time_s = os.getenv("ASOF_TIME", "09:40").strip()
    cutoff = pd.Timestamp(f"{date_s} {time_s}:00")

    full_day = fetch_5m(symbol, date_s)

    # 关键约束：所有计算之前先截断到模拟时点；后续K线不允许进入任何特征或状态计算。
    prefix = full_day[full_day["ts"] <= cutoff].copy().sort_values("ts")
    if prefix.empty:
        raise RuntimeError(f"{date_s} {time_s} 之前没有完整5分钟K线")

    last = prefix.iloc[-1]
    last_ts = pd.Timestamp(last["ts"])
    current = float(last["close"])
    running_high = float(prefix["high"].max())
    running_low = float(prefix["low"].min())

    # V3.7当前正式验证范围从09:50开始；09:40不能强行输出概率。
    eligible = last_ts.time() >= datetime.strptime("09:50", "%H:%M").time()
    if not eligible:
        top_text = "暂不判断"
        bottom_text = "暂不判断"
        conclusion = "等待09:50首个有效判断"
        status = "等待"
        color = "blue"
    else:
        # 此脚本当前只负责严格as-of数据重放与早盘门控；正式概率接入由后续live scorer完成。
        top_text = "待模型评分"
        bottom_text = "待模型评分"
        conclusion = "已到有效时点，等待V3.7评分器"
        status = "观察"
        color = "blue"

    result = {
        "证券名称": SYMBOL_NAMES.get(symbol, symbol),
        "证券代码": symbol,
        "模拟时间": f"{date_s} {time_s}",
        "实际使用到的最后K线": last_ts.strftime("%Y-%m-%d %H:%M"),
        "当前价格": round(current, 2),
        "日内高点": round(running_high, 2),
        "日内低点": round(running_low, 2),
        "顶部判断": top_text,
        "底部判断": bottom_text,
        "当前状态": status,
        "操作结论": conclusion,
        "严格截至时点": True,
        "使用K线根数": int(len(prefix)),
        "未来K线参与计算": False,
    }
    (RESULTS / "asof_replay_v37.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    card = {
        "卡片标题": f"{SYMBOL_NAMES.get(symbol, symbol)}｜{time_s}｜{status}",
        "标题颜色": color,
        "时间": time_s,
        "现价": f"{current:.2f}",
        "顶部": top_text,
        "底部": bottom_text,
        "结论": conclusion,
        "数据性质": "历史时点重放",
    }
    (RESULTS / "asof_replay_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ASOF_REPLAY_OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
