import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import backtest_v13 as v13
import extreme_running_lock_walkforward_v37 as v37
from asof_replay_v37 import append_prefix, build_current_feature_row
from backtest_v14_cloud import load_a_cloud
from live_monitor_v37 import (
    INDEX,
    NAME,
    SYMBOL,
    fetch_tick,
    load_models,
    send_compact,
    tick_times,
)

TZ = ZoneInfo("Asia/Shanghai")


def latest_settled_target(now):
    settled = [x for x in tick_times(now.date()) if (x.timestamp() + 25) <= now.timestamp()]
    return settled[-1] if settled else None


def main():
    now = datetime.now(TZ)
    target_dt = latest_settled_target(now)
    if target_dt is None:
        print("LIVE_TICK_OUTSIDE_WINDOW", now.isoformat(), flush=True)
        return

    # Do not run during lunch after the morning close or after 15:05.
    hhmm = now.strftime("%H:%M")
    if "11:31" <= hhmm < "13:05" or hhmm > "15:10":
        print("LIVE_TICK_OUTSIDE_WINDOW", now.isoformat(), flush=True)
        return

    requested_cutoff = pd.Timestamp(target_dt.replace(tzinfo=None))
    print("LIVE_TICK_START", requested_cutoff, flush=True)

    v13.load_a = load_a_cloud
    top_model, bottom_model, train_end = load_models(now.date())

    prefixes, idx_prefix, source, cutoff = fetch_tick(requested_cutoff)
    for s, p in prefixes.items():
        append_prefix(s, p, False)
    append_prefix(INDEX, idx_prefix, True)

    prefix = prefixes[SYMBOL]
    last = prefix.iloc[-1]
    current = float(last.close)
    running_high = float(prefix.high.max())
    running_low = float(prefix.low.min())
    bar_time = cutoff.strftime("%H:%M")

    if bar_time < "09:50":
        send_compact(
            f"{NAME}｜{bar_time}｜等待",
            "blue",
            bar_time,
            current,
            "—",
            "—",
            "模型有效时点从09:50开始",
        )
        print("LIVE_TICK_WAIT", bar_time, flush=True)
        return

    row = build_current_feature_row(cutoff)
    if pd.Timestamp(row.iloc[-1].ts) > cutoff:
        raise RuntimeError("未来K线泄漏保护触发")

    p_top = float(v37.predict(top_model, row)[0])
    p_bottom = float(v37.predict(bottom_model, row)[0])
    gap_high = max(0.0, (running_high - current) / current)
    gap_low = max(0.0, (current - running_low) / current)
    near_high = gap_high <= v37.NEAR_EXTREME_MAX
    near_low = gap_low <= v37.NEAR_EXTREME_MAX

    if p_top > p_bottom and near_high:
        state, conclusion, color = "反T观察", "顶部概率占优，反T观察", "orange"
    elif p_bottom > p_top and near_low:
        state, conclusion, color = "正T观察", "底部概率占优，正T观察", "green"
    elif p_top > p_bottom:
        state, conclusion, color = "观察", "顶部概率占优，但已离高点", "blue"
    else:
        state, conclusion, color = "观察", "底部概率占优，但已离低点", "blue"

    rec = {
        "请求时间": str(requested_cutoff),
        "实际共同K线": str(cutoff),
        "现价": round(current, 2),
        "顶部锁定概率": round(p_top, 6),
        "底部锁定概率": round(p_bottom, 6),
        "距日高": round(gap_high, 6),
        "距日低": round(gap_low, 6),
        "状态": state,
        "结论": conclusion,
        "行情源": source,
        "训练截止": train_end,
        "未来K线参与计算": False,
    }
    print("LIVE_SCORE", json.dumps(rec, ensure_ascii=False), flush=True)
    send_compact(
        f"{NAME}｜{bar_time}｜{state}",
        color,
        bar_time,
        current,
        f"{p_top * 100:.0f}%",
        f"{p_bottom * 100:.0f}%",
        conclusion,
    )
    print("LIVE_TICK_DONE", bar_time, flush=True)


if __name__ == "__main__":
    main()
