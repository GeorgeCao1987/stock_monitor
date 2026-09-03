import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pytdx.hq import TdxHq_API

import backtest_v13 as v13
import event_engine_v14 as e14
import extreme_pattern_mining_v30 as v30
import t_edge_daily_regime_model_v29 as v29
import extreme_running_lock_walkforward_v37 as v37
from backtest_v14_cloud import load_a_cloud
from oos_core_collect import PCB_STOCKS, INDEX, server_list

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
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


def fetch_day_5m(symbol: str, target_date: str, is_index: bool = False) -> pd.DataFrame:
    market = 1 if (is_index or symbol.startswith("6")) else 0
    code = "000001" if is_index else symbol
    target = pd.Timestamp(target_date)
    for ip, port in server_list():
        api = None
        try:
            api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
            if not api.connect(ip, port, time_out=2):
                continue
            getter = api.get_index_bars if is_index else api.get_security_bars
            parts = []
            for page in range(6):
                rows = getter(0, market, code, page * 800, 800) or []
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
            if z.empty:
                continue
            for c in ["open", "high", "low", "close", "vol", "amount"]:
                if c in z.columns:
                    z[c] = pd.to_numeric(z[c], errors="coerce")
            out = pd.DataFrame({
                "ts": z.ts,
                "open": z.open,
                "high": z.high,
                "low": z.low,
                "close": z.close,
                "volume": z["vol"] if "vol" in z.columns else 0,
                "amount": z["amount"] if "amount" in z.columns else np.nan,
                "source": "pytdx_index" if is_index else "pytdx_stock",
            }).dropna(subset=["ts", "close"])
            print("ASOF_DAY_SOURCE_OK", symbol, ip, port, len(out))
            return out
        except Exception as exc:
            print("ASOF_DAY_SERVER_FAIL", symbol, ip, port, repr(exc))
        finally:
            if api is not None:
                try:
                    api.disconnect()
                except Exception:
                    pass
    raise RuntimeError(f"无法取得 {symbol} {target_date} 的5分钟行情")


def append_prefix(symbol: str, prefix: pd.DataFrame, is_index: bool = False) -> None:
    key = symbol.replace(".", "_")
    src = "pytdx_index" if is_index else "pytdx_stock"
    path = DATA / f"a_{key}_{src}.csv"
    if not path.exists():
        raise RuntimeError(f"历史训练数据不存在: {path}")
    hist = pd.read_csv(path)
    hist["ts"] = pd.to_datetime(hist["ts"], errors="coerce")
    z = pd.concat([hist, prefix], ignore_index=True)
    z["ts"] = pd.to_datetime(z["ts"], errors="coerce")
    z = z.dropna(subset=["ts", "close"]).drop_duplicates("ts", keep="last").sort_values("ts")
    z.to_csv(path, index=False)


def build_current_feature_row(cutoff: pd.Timestamp) -> pd.DataFrame:
    # All functions below are causal for the feature columns used by V3.7.
    # The target day's files have already been physically truncated to cutoff.
    x = e14.build_scored_frame()
    x = v30.add_past_features(x)
    d = v29.daily_context()
    x = x.merge(d, on="day", how="left")

    x["gap_high"] = ((x.cum_high - x.close) / x.close).clip(lower=0)
    x["gap_low"] = ((x.close - x.cum_low) / x.close).clip(lower=0)
    atr = pd.to_numeric(x.atr_pct, errors="coerce").replace(0, np.nan)
    x["gap_high_atr"] = x.gap_high / atr
    x["gap_low_atr"] = x.gap_low / atr

    row = x[x.ts == cutoff].copy()
    if row.empty:
        # Be conservative if a vendor timestamps a completed bar a few seconds differently.
        row = x[(x.ts.dt.date == cutoff.date()) & (x.ts <= cutoff)].sort_values("ts").tail(1).copy()
    if row.empty:
        raise RuntimeError(f"无法构造 {cutoff} 的模型特征")
    return row


def probability_text(p: float) -> str:
    return f"{p * 100:.0f}%"


def main():
    symbol = os.getenv("ASOF_SYMBOL", "002916").strip()
    date_s = os.getenv("ASOF_DATE", "2026-09-03").strip()
    time_s = os.getenv("ASOF_TIME", "09:40").strip()
    cutoff = pd.Timestamp(f"{date_s} {time_s}:00")

    # Training history has already been collected only through the prior calendar day.
    # Fetch the target day separately and physically truncate every domestic series before
    # appending it to the feature data store. Thus 10:05+ bars cannot enter feature code.
    prefixes = {}
    for s in PCB_STOCKS:
        full = fetch_day_5m(s, date_s, is_index=False)
        p = full[full.ts <= cutoff].copy().sort_values("ts")
        if p.empty:
            raise RuntimeError(f"{s} 在 {cutoff} 前无完整5分钟K线")
        prefixes[s] = p
        append_prefix(s, p, is_index=False)

    idx_full = fetch_day_5m(INDEX, date_s, is_index=True)
    idx_prefix = idx_full[idx_full.ts <= cutoff].copy().sort_values("ts")
    if idx_prefix.empty:
        raise RuntimeError(f"{INDEX} 在 {cutoff} 前无完整5分钟K线")
    append_prefix(INDEX, idx_prefix, is_index=True)

    if symbol not in prefixes:
        raise RuntimeError(f"暂不支持标的 {symbol}")
    prefix = prefixes[symbol]
    last = prefix.iloc[-1]
    last_ts = pd.Timestamp(last.ts)
    current = float(last.close)
    running_high = float(prefix.high.max())
    running_low = float(prefix.low.min())

    eligible_time = last_ts.time() >= datetime.strptime("09:50", "%H:%M").time()

    # Cloud files use PyTDX source names chosen from completeness.csv.
    v13.load_a = load_a_cloud

    if not eligible_time:
        top_text = "暂不判断"
        bottom_text = "暂不判断"
        conclusion = "等待09:50首个有效判断"
        status = "等待"
        color = "blue"
        p_top = None
        p_bottom = None
    else:
        # Training labels are built only from complete days strictly before target date.
        raw = v37.build_frame()
        target_day = cutoff.date()
        train = raw[(raw.day >= v37.BULL_START) & (raw.day < target_day) & v37.eligible(raw)].copy()
        if train.day.nunique() < 60:
            raise RuntimeError(f"训练样本不足: {train.day.nunique()} days")

        row = build_current_feature_row(cutoff)
        if pd.Timestamp(row.iloc[-1].ts) > cutoff:
            raise RuntimeError("未来K线泄漏保护触发")

        p_top = float(v37.predict(v37.fit_model(train, "top_locked_running"), row)[0])
        p_bottom = float(v37.predict(v37.fit_model(train, "bottom_locked_running"), row)[0])
        top_text = probability_text(p_top)
        bottom_text = probability_text(p_bottom)

        gap_high = max(0.0, (running_high - current) / current)
        gap_low = max(0.0, (current - running_low) / current)
        near_high = gap_high <= v37.NEAR_EXTREME_MAX
        near_low = gap_low <= v37.NEAR_EXTREME_MAX

        # V3.7 threshold remains diagnostic; conclusions are WATCH only, never EXECUTE.
        if p_top > p_bottom and near_high:
            conclusion = "顶部概率占优，反T观察"
            status = "反T观察"
            color = "orange"
        elif p_bottom > p_top and near_low:
            conclusion = "底部概率占优，正T观察"
            status = "正T观察"
            color = "green"
        elif p_top > p_bottom:
            conclusion = "顶部概率占优，但已离高点"
            status = "观察"
            color = "blue"
        else:
            conclusion = "底部概率占优，但已离低点"
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
        "顶部锁定概率": None if p_top is None else round(p_top, 6),
        "底部锁定概率": None if p_bottom is None else round(p_bottom, 6),
        "顶部判断": top_text,
        "底部判断": bottom_text,
        "当前状态": status,
        "操作结论": conclusion,
        "严格截至时点": True,
        "使用K线根数": int(len(prefix)),
        "训练数据截止": os.getenv("MARKET_END_DATE", ""),
        "未来K线参与计算": False,
        "阈值状态": "V3.7诊断阈值，当前只输出观察，不输出执行",
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
        "数据性质": "严格历史时点重放",
    }
    (RESULTS / "asof_replay_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ASOF_REPLAY_OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
