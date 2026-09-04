import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from pytdx.hq import TdxHq_API

import backtest_v13 as v13
import extreme_running_lock_walkforward_v37 as v37
from asof_replay_v37 import append_prefix, build_current_feature_row
from backtest_v14_cloud import load_a_cloud
from feishu_notifier import build_signal_card, send_card
from oos_core_collect import PCB_STOCKS, INDEX, server_list

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
AUDIT = RESULTS / "live_monitor_v37.jsonl"
TZ = ZoneInfo("Asia/Shanghai")
SYMBOL = os.getenv("LIVE_SYMBOL", "002916").strip()
NAME = "深南电路" if SYMBOL == "002916" else SYMBOL
PUSH_DELTA = float(os.getenv("LIVE_PUSH_DELTA", "0.15"))
BAR_SETTLE_SECONDS = int(os.getenv("LIVE_BAR_SETTLE_SECONDS", "25"))
MODEL_BUNDLE = BASE / "live_cache" / "model_bundle.joblib"
MAX_COMMON_LAG_MINUTES = int(os.getenv("LIVE_MAX_COMMON_LAG_MINUTES", "5"))

PREFERRED = [("202.108.253.139", 80)]


def servers():
    out = []
    for item in PREFERRED + list(server_list()):
        if item not in out:
            out.append(item)
    return out


def parse_rows(api, rows, target_date, is_index=False):
    if not rows:
        return pd.DataFrame()
    z = api.to_df(rows)
    z["ts"] = pd.to_datetime(z["datetime"], errors="coerce")
    z = z.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts")
    z = z[z.ts.dt.strftime("%Y-%m-%d") == target_date].copy()
    if z.empty:
        return z
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    return pd.DataFrame({
        "ts": z.ts,
        "open": z.open,
        "high": z.high,
        "low": z.low,
        "close": z.close,
        "volume": z["vol"] if "vol" in z.columns else 0,
        "amount": z["amount"] if "amount" in z.columns else np.nan,
        "source": "pytdx_index" if is_index else "pytdx_stock",
    }).dropna(subset=["ts", "close"])


def _latest_common_cutoff(series_map, requested_cutoff):
    timestamp_sets = []
    for df in series_map.values():
        q = df[df.ts <= requested_cutoff]
        if q.empty:
            return None
        timestamp_sets.append(set(pd.to_datetime(q.ts)))
    common = set.intersection(*timestamp_sets) if timestamp_sets else set()
    if not common:
        return None
    return max(common)


def fetch_tick(requested_cutoff: pd.Timestamp):
    date_s = requested_cutoff.strftime("%Y-%m-%d")
    last_error = None
    for attempt in range(4):
        for ip, port in servers():
            api = None
            try:
                api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
                if not api.connect(ip, port, time_out=1.5):
                    continue

                full_map = {}
                for s in PCB_STOCKS:
                    rows = api.get_security_bars(0, 1 if s.startswith("6") else 0, s, 0, 800) or []
                    full_map[s] = parse_rows(api, rows, date_s, False)
                rows = api.get_index_bars(0, 1, "000001", 0, 800) or []
                full_map[INDEX] = parse_rows(api, rows, date_s, True)

                common_cutoff = _latest_common_cutoff(full_map, requested_cutoff)
                if common_cutoff is None:
                    last_error = RuntimeError("7条行情不存在共同完整5分钟K线")
                    continue

                lag_minutes = (requested_cutoff - common_cutoff).total_seconds() / 60.0
                if lag_minutes > MAX_COMMON_LAG_MINUTES + 1e-9:
                    last_error = RuntimeError(
                        f"共同K线过旧: requested={requested_cutoff}, common={common_cutoff}, lag={lag_minutes:.1f}m"
                    )
                    continue

                prefixes = {}
                for s in PCB_STOCKS:
                    prefixes[s] = full_map[s][full_map[s].ts <= common_cutoff].copy().sort_values("ts")
                idx_prefix = full_map[INDEX][full_map[INDEX].ts <= common_cutoff].copy().sort_values("ts")

                return prefixes, idx_prefix, f"{ip}:{port}", pd.Timestamp(common_cutoff)
            except Exception as exc:
                last_error = exc
            finally:
                if api is not None:
                    try:
                        api.disconnect()
                    except Exception:
                        pass
        if attempt < 3:
            time.sleep(10)
    raise RuntimeError(f"实时行情获取失败: {last_error!r}")


def send_compact(title, color, bar_time, price, top, bottom, conclusion):
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    if not webhook:
        raise RuntimeError("FEISHU_WEBHOOK_URL missing")
    data = {
        "卡片标题": title,
        "标题颜色": color,
        "时间": bar_time,
        "现价": "—" if price is None else f"{price:.2f}",
        "顶部": top,
        "底部": bottom,
        "结论": conclusion,
    }
    result = send_card(webhook, build_signal_card(data))
    print("FEISHU_LIVE_PUSH_OK", json.dumps(result, ensure_ascii=False), flush=True)


def write_audit(rec):
    with AUDIT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def tick_times(day):
    out = []
    t = datetime.combine(day, datetime.strptime("09:35", "%H:%M").time(), TZ)
    end = datetime.combine(day, datetime.strptime("11:30", "%H:%M").time(), TZ)
    while t <= end:
        out.append(t)
        t += timedelta(minutes=5)
    t = datetime.combine(day, datetime.strptime("13:05", "%H:%M").time(), TZ)
    end = datetime.combine(day, datetime.strptime("15:00", "%H:%M").time(), TZ)
    while t <= end:
        out.append(t)
        t += timedelta(minutes=5)
    return out


def wait_until(dt):
    while True:
        now = datetime.now(TZ)
        remain = (dt - now).total_seconds()
        if remain <= 0:
            return
        time.sleep(min(30, remain))


def startup_pending_times(times):
    now = datetime.now(TZ)
    settled = [x for x in times if x + timedelta(seconds=BAR_SETTLE_SECONDS) <= now]
    future = [x for x in times if x + timedelta(seconds=BAR_SETTLE_SECONDS) > now]
    pending = []
    if settled:
        pending.append(settled[-1])
    pending.extend(future)
    return pending


def load_models(today):
    if MODEL_BUNDLE.exists():
        bundle = joblib.load(MODEL_BUNDLE)
        print(
            "LIVE_MODEL_CACHE_READY",
            bundle.get("train_days"), bundle.get("train_start"), bundle.get("train_end"),
            flush=True,
        )
        return bundle["top_model"], bundle["bottom_model"], str(bundle.get("train_end", ""))

    print("LIVE_MODEL_CACHE_MISS_FALLBACK_TRAIN", flush=True)
    v13.load_a = load_a_cloud
    raw = v37.build_frame()
    train = raw[(raw.day >= v37.BULL_START) & (raw.day < today) & v37.eligible(raw)].copy()
    if train.day.nunique() < 60:
        raise RuntimeError(f"训练样本不足: {train.day.nunique()} days")
    top_model = v37.fit_model(train, "top_locked_running")
    bottom_model = v37.fit_model(train, "bottom_locked_running")
    return top_model, bottom_model, str(max(train.day))


def main():
    today = datetime.now(TZ).date()
    print("LIVE_MONITOR_START", today, flush=True)
    v13.load_a = load_a_cloud
    top_model, bottom_model, train_end = load_models(today)

    times = tick_times(today)
    pending = startup_pending_times(times)
    if not pending:
        print("LIVE_MONITOR_OUTSIDE_WINDOW", flush=True)
        return

    startup_sent = False
    first_score_sent = False
    last_state = None
    last_push_top = None
    last_push_bottom = None
    last_scored_cutoff = None

    for target_dt in pending:
        wait_until(target_dt + timedelta(seconds=BAR_SETTLE_SECONDS))
        requested_cutoff = pd.Timestamp(target_dt.replace(tzinfo=None))
        requested_bar_time = target_dt.strftime("%H:%M")
        try:
            prefixes, idx_prefix, source, cutoff = fetch_tick(requested_cutoff)
            if last_scored_cutoff is not None and cutoff <= last_scored_cutoff:
                print(
                    "LIVE_DUPLICATE_COMMON_BAR_SKIP",
                    requested_bar_time, cutoff.strftime("%H:%M"),
                    flush=True,
                )
                continue

            for s, p in prefixes.items():
                append_prefix(s, p, False)
            append_prefix(INDEX, idx_prefix, True)

            prefix = prefixes[SYMBOL]
            last = prefix.iloc[-1]
            current = float(last.close)
            running_high = float(prefix.high.max())
            running_low = float(prefix.low.min())
            bar_time = cutoff.strftime("%H:%M")

            if not startup_sent:
                send_compact(
                    f"{NAME}｜实时监控已启动", "blue", bar_time, current, "—", "—",
                    "实时行情、模型与飞书链路正常",
                )
                startup_sent = True

            if bar_time < "09:50":
                write_audit({
                    "请求时间": str(requested_cutoff), "实际共同K线": str(cutoff),
                    "现价": current, "状态": "等待有效时点",
                    "行情源": source, "未来K线参与计算": False,
                })
                last_scored_cutoff = cutoff
                continue

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
                "请求时间": str(requested_cutoff), "实际共同K线": str(cutoff),
                "现价": round(current, 2),
                "顶部锁定概率": round(p_top, 6), "底部锁定概率": round(p_bottom, 6),
                "距日高": round(gap_high, 6), "距日低": round(gap_low, 6),
                "状态": state, "结论": conclusion, "行情源": source,
                "训练截止": train_end, "未来K线参与计算": False,
            }
            write_audit(rec)
            print("LIVE_SCORE", json.dumps(rec, ensure_ascii=False), flush=True)

            changed = last_state is None or state != last_state
            moved = (
                last_push_top is None
                or abs(p_top - last_push_top) >= PUSH_DELTA
                or abs(p_bottom - last_push_bottom) >= PUSH_DELTA
            )
            should_push = (not first_score_sent) or changed or moved
            if should_push:
                send_compact(
                    f"{NAME}｜{bar_time}｜{state}", color, bar_time, current,
                    f"{p_top * 100:.0f}%", f"{p_bottom * 100:.0f}%", conclusion,
                )
                first_score_sent = True
                last_state = state
                last_push_top = p_top
                last_push_bottom = p_bottom
            last_scored_cutoff = cutoff
        except Exception as exc:
            rec = {"请求时间": str(requested_cutoff), "错误": repr(exc), "未来K线参与计算": False}
            write_audit(rec)
            print("LIVE_TICK_FAIL", json.dumps(rec, ensure_ascii=False), flush=True)
            try:
                send_compact(
                    f"{NAME}｜{requested_bar_time}｜数据异常", "red", requested_bar_time, None, "—", "—",
                    "本周期未计算，请勿使用旧信号",
                )
            except Exception as push_exc:
                print("LIVE_ERROR_PUSH_FAIL", repr(push_exc), flush=True)

    print("LIVE_MONITOR_END", flush=True)


if __name__ == "__main__":
    main()
