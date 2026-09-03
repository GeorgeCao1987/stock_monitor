from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

PCB = ["002916", "002463", "600183", "002938", "603228", "300476"]
TARGET = "002916"
INDEX_FILE = "a_000001_SH_pytdx_index.csv"
OPEN_BARS = 6

DEV_PERIODS = {
    "jan_feb": ("2026-01-01", "2026-02-28"),
    "mar_apr": ("2026-03-01", "2026-04-30"),
    "may_jun": ("2026-05-01", "2026-06-30"),
}
VALID_PERIOD = {"jul_aug": ("2026-07-01", "2026-08-31")}


def _load(path):
    z = pd.read_csv(path)
    z["ts"] = pd.to_datetime(z["ts"])
    z["day"] = z["ts"].dt.date
    return z.sort_values("ts")


def _daily_from_5m(z):
    rows = []
    for day, g in z.groupby("day", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) != 48:
            continue
        rows.append({
            "day": day,
            "open": float(g.iloc[0].open),
            "close": float(g.iloc[-1].close),
            "close_10": float(g.iloc[OPEN_BARS - 1].close),
            "high_bar_no": int(g.high.to_numpy().argmax()),
            "low_bar_no": int(g.low.to_numpy().argmin()),
            "first_30_high": float(g.iloc[:OPEN_BARS].high.max()),
            "first_30_low": float(g.iloc[:OPEN_BARS].low.min()),
        })
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d["prev_close"] = d["close"].shift(1)
    d["gap"] = d["open"] / d["prev_close"] - 1.0
    return d


def build_daily():
    daily = {}
    for sym in PCB:
        daily[sym] = _daily_from_5m(_load(DATA / f"a_{sym}_pytdx_stock.csv"))
    idx = _daily_from_5m(_load(DATA / INDEX_FILE))

    t = daily[TARGET][[
        "day", "open", "close", "close_10", "high_bar_no", "low_bar_no", "gap"
    ]].rename(columns={"gap": "target_gap"})
    t["high_after_10"] = t["high_bar_no"] >= OPEN_BARS
    t["low_after_10"] = t["low_bar_no"] >= OPEN_BARS
    t["ret_0925_1000"] = t["close_10"] / t["open"] - 1.0
    t["final_ret_from_open"] = t["close"] / t["open"] - 1.0

    gaps = []
    for sym in PCB:
        q = daily[sym][["day", "gap"]].rename(columns={"gap": f"gap_{sym}"})
        gaps.append(q)
    for q in gaps:
        t = t.merge(q, on="day", how="inner")
    t = t.merge(idx[["day", "gap"]].rename(columns={"gap": "index_gap"}), on="day", how="inner")

    gap_cols = [f"gap_{s}" for s in PCB]
    t["pcb_gap_mean"] = t[gap_cols].mean(axis=1)
    t["pcb_gap_median"] = t[gap_cols].median(axis=1)
    t["pcb_up_breadth"] = (t[gap_cols] > 0).mean(axis=1)
    t["pcb_down_breadth"] = (t[gap_cols] < 0).mean(axis=1)
    t["pcb_rel_gap"] = t["pcb_gap_mean"] - t["index_gap"]
    t["target_rel_gap"] = t["target_gap"] - t["index_gap"]
    t = t.dropna(subset=["target_gap", "index_gap", "pcb_gap_mean"]).copy()
    t["day"] = pd.to_datetime(t["day"])
    return t


def apply_rule(z, p):
    up = (
        (z.pcb_gap_mean >= p["mean_thr"]).astype(int)
        + (z.pcb_up_breadth >= p["breadth_thr"]).astype(int)
        + (z.pcb_rel_gap >= p["rel_thr"]).astype(int)
        + (z.target_gap >= p["target_thr"]).astype(int)
    )
    down = (
        (z.pcb_gap_mean <= -p["mean_thr"]).astype(int)
        + (z.pcb_down_breadth >= p["breadth_thr"]).astype(int)
        + (z.pcb_rel_gap <= -p["rel_thr"]).astype(int)
        + (z.target_gap <= -p["target_thr"]).astype(int)
    )
    state = np.select(
        [
            (up >= p["votes"]) & ((up - down) >= p["margin"]),
            (down >= p["votes"]) & ((down - up) >= p["margin"]),
        ],
        ["PREOPEN_BULLISH", "PREOPEN_BEARISH"],
        default="PREOPEN_NEUTRAL",
    )
    return pd.Series(state, index=z.index)


def metrics(z, state_col="state"):
    n = len(z)
    bull = z[z[state_col] == "PREOPEN_BULLISH"]
    bear = z[z[state_col] == "PREOPEN_BEARISH"]
    neutral = z[z[state_col] == "PREOPEN_NEUTRAL"]
    return {
        "days": int(n),
        "bull_signals": int(len(bull)),
        "bear_signals": int(len(bear)),
        "neutral_days": int(len(neutral)),
        "bull_coverage": float(len(bull) / n) if n else None,
        "bear_coverage": float(len(bear) / n) if n else None,
        "bull_high_after_10_win_rate": float(bull.high_after_10.mean()) if len(bull) else None,
        "bear_low_after_10_win_rate": float(bear.low_after_10.mean()) if len(bear) else None,
        "bull_0925_1000_direction_win_rate": float((bull.ret_0925_1000 > 0).mean()) if len(bull) else None,
        "bear_0925_1000_direction_win_rate": float((bear.ret_0925_1000 < 0).mean()) if len(bear) else None,
        "bull_median_0925_1000_ret": float(bull.ret_0925_1000.median()) if len(bull) else None,
        "bear_median_0925_1000_ret": float(bear.ret_0925_1000.median()) if len(bear) else None,
        "bull_median_final_ret": float(bull.final_ret_from_open.median()) if len(bull) else None,
        "bear_median_final_ret": float(bear.final_ret_from_open.median()) if len(bear) else None,
    }


def slice_period(z, start, end):
    return z[(z.day >= pd.Timestamp(start)) & (z.day <= pd.Timestamp(end))].copy()


def choose_rule(z):
    candidates = []
    grid = itertools.product(
        [0.001, 0.002, 0.003, 0.005],
        [0.50, 2/3, 5/6],
        [0.0, 0.001, 0.002, 0.003],
        [0.002, 0.004, 0.006, 0.010],
        [2, 3, 4],
        [1, 2],
    )
    for mean_thr, breadth_thr, rel_thr, target_thr, votes, margin in grid:
        p = {
            "mean_thr": mean_thr,
            "breadth_thr": breadth_thr,
            "rel_thr": rel_thr,
            "target_thr": target_thr,
            "votes": votes,
            "margin": margin,
        }
        per = {}
        good = True
        win_rates = []
        counts = []
        for name, (start, end) in DEV_PERIODS.items():
            q = slice_period(z, start, end)
            q["state"] = apply_rule(q, p)
            m = metrics(q)
            per[name] = m
            if m["bull_signals"] < 4 or m["bear_signals"] < 4:
                good = False
                break
            win_rates.extend([m["bull_high_after_10_win_rate"], m["bear_low_after_10_win_rate"]])
            counts.extend([m["bull_signals"], m["bear_signals"]])
        if not good:
            continue
        min_win = min(win_rates)
        mean_win = float(np.mean(win_rates))
        dev_q = slice_period(z, "2026-01-01", "2026-06-30")
        dev_q["state"] = apply_rule(dev_q, p)
        dm = metrics(dev_q)
        if dm["bull_signals"] < 20 or dm["bear_signals"] < 20:
            continue
        coverage = dm["bull_coverage"] + dm["bear_coverage"]
        candidates.append((min_win, mean_win, coverage, p, per, dm))
    if not candidates:
        raise RuntimeError("no auction rule met minimum sample constraints")
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0], candidates[:20]


def main():
    z = build_daily()
    baseline = {
        "days": int(len(z)),
        "unconditional_high_after_10": float(z.high_after_10.mean()),
        "unconditional_low_after_10": float(z.low_after_10.mean()),
        "unconditional_0925_1000_up": float((z.ret_0925_1000 > 0).mean()),
        "unconditional_0925_1000_down": float((z.ret_0925_1000 < 0).mean()),
    }

    best, top20 = choose_rule(z)
    min_win, mean_win, coverage, rule, dev_period_metrics, dev_metrics = best

    qv = slice_period(z, *VALID_PERIOD["jul_aug"])
    qv["state"] = apply_rule(qv, rule)
    valid_metrics = metrics(qv)

    z["state"] = apply_rule(z, rule)
    period_metrics = {}
    for name, bounds in {**DEV_PERIODS, **VALID_PERIOD}.items():
        period_metrics[name] = metrics(slice_period(z, *bounds))

    report = {
        "version": "auction-final-diagnostics-v21",
        "status": "development_rule_selected_on_2026_01_06_validated_on_2026_07_08",
        "data_note": (
            "This backtest uses the first 5-minute bar open as the 09:25 final call-auction price. "
            "It does NOT contain historical 09:15/09:20/09:24 virtual-match trajectory or unmatched volume."
        ),
        "objective": {
            "PREOPEN_BULLISH_win": "true daily high occurs after 10:00",
            "PREOPEN_BEARISH_win": "true daily low occurs after 10:00",
            "secondary": "09:25 to 10:00 price direction",
        },
        "no_future_leakage": True,
        "target": TARGET,
        "pcb_basket": PCB,
        "baseline": baseline,
        "selected_rule": rule,
        "selection_score": {
            "minimum_dev_period_side_win_rate": min_win,
            "mean_dev_period_side_win_rate": mean_win,
            "dev_total_signal_coverage": coverage,
        },
        "development_2026_01_06": dev_metrics,
        "validation_2026_07_08": valid_metrics,
        "by_period": period_metrics,
        "top20_rules": [
            {
                "min_win": a,
                "mean_win": b,
                "coverage": c,
                "rule": p,
            }
            for a, b, c, p, _, _ in top20
        ],
        "important_limitation": (
            "Full auction-process V2 requires historical 09:15-09:25 snapshots. "
            "This result validates only the final 09:25 auction-price/breadth layer."
        ),
    }

    z.to_csv(RESULTS / "auction_final_daily_v21.csv", index=False)
    (RESULTS / "auction_final_diagnostics_v21.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("AUCTION FINAL DIAGNOSTICS V2.1")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
