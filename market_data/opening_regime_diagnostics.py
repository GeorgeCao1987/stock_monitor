from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import event_engine_v17 as e17

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

OPEN_BARS = 6  # 09:35..10:00 for standard A-share 5m bars


def _opening_snapshot(g: pd.DataFrame) -> dict:
    g = g.sort_values("ts").reset_index(drop=True)
    first = g.iloc[:OPEN_BARS].copy()
    if len(first) < OPEN_BARS:
        return {"opening_ready": False, "opening_regime": "PENDING"}

    r = first.iloc[-1]
    hi = float(first.high.max())
    lo = float(first.low.min())
    rng = hi - lo
    close_pos = (float(r.close) - lo) / rng if rng > 0 else .5
    above_vwap_share = float((first.close > first.vwap).mean())

    up = 0
    down = 0
    up += int(r.ret_from_open >= .004)
    down += int(r.ret_from_open <= -.004)
    up += int(close_pos >= .65)
    down += int(close_pos <= .35)
    up += int(r.dist_vwap >= .0015)
    down += int(r.dist_vwap <= -.0015)

    if pd.notna(r.get("pcb_rel")):
        up += int(r.pcb_rel >= .0010)
        down += int(r.pcb_rel <= -.0010)
    if pd.notna(r.get("pcb_up_breadth")):
        up += int(r.pcb_up_breadth >= .60)
        down += int(r.pcb_up_breadth <= .40)
    if pd.notna(r.get("index_ret")):
        up += int(r.index_ret >= .0020)
        down += int(r.index_ret <= -.0020)

    # Overseas priors are used only when present. Missing feeds never create a vote.
    overseas_up = 0
    overseas_down = 0
    for col in ["hynix_ret", "samsung_ret", "kospi_ret"]:
        v = r.get(col)
        if pd.notna(v):
            overseas_up += int(v > 0)
            overseas_down += int(v < 0)
    sox = r.get("sox_prev_daily_ret")
    if pd.notna(sox):
        overseas_up += int(sox > 0)
        overseas_down += int(sox < 0)
    if overseas_up >= 2 and overseas_up > overseas_down:
        up += 1
    if overseas_down >= 2 and overseas_down > overseas_up:
        down += 1

    if up >= 4 and up - down >= 2:
        regime = "UP"
    elif down >= 4 and down - up >= 2:
        regime = "DOWN"
    else:
        regime = "MIXED"

    return {
        "opening_ready": True,
        "opening_regime": regime,
        "opening_up_votes": int(up),
        "opening_down_votes": int(down),
        "opening_ret": float(r.ret_from_open),
        "opening_dist_vwap": float(r.dist_vwap),
        "opening_close_pos": float(close_pos),
        "opening_above_vwap_share": above_vwap_share,
        "opening_pcb_rel": float(r.pcb_rel) if pd.notna(r.get("pcb_rel")) else None,
        "opening_pcb_breadth": float(r.pcb_up_breadth) if pd.notna(r.get("pcb_up_breadth")) else None,
        "opening_index_ret": float(r.index_ret) if pd.notna(r.get("index_ret")) else None,
        "opening_overseas_votes_present": int(overseas_up + overseas_down),
    }


def add_opening_regime(x: pd.DataFrame):
    z = x.copy().sort_values("ts").reset_index(drop=True)
    z["day"] = z.ts.dt.date
    z["bar_no"] = z.groupby("day").cumcount()

    rows = []
    for day, g in z.groupby("day", sort=True):
        rec = {"day": day}
        rec.update(_opening_snapshot(g))

        gg = g.sort_values("ts").reset_index(drop=True)
        final = gg.iloc[-1]
        day_hi = float(gg.high.max())
        day_lo = float(gg.low.min())
        day_rng = day_hi - day_lo
        final_pos = (float(final.close) - day_lo) / day_rng if day_rng > 0 else .5
        final_ret = float(final.close / gg.iloc[0].open - 1)
        if final_ret >= .005 and final_pos >= .65:
            truth = "UP"
        elif final_ret <= -.005 and final_pos <= .35:
            truth = "DOWN"
        else:
            truth = "MIXED"

        rec.update({
            "truth_day_type": truth,
            "final_ret": final_ret,
            "final_pos_in_day_range": float(final_pos),
            "high_bar_no": int(gg.high.to_numpy().argmax()),
            "low_bar_no": int(gg.low.to_numpy().argmin()),
        })
        rows.append(rec)

    daily = pd.DataFrame(rows)
    z = z.merge(daily[[
        "day", "opening_regime", "opening_up_votes", "opening_down_votes",
        "opening_ret", "opening_dist_vwap", "opening_close_pos",
        "opening_above_vwap_share", "opening_pcb_rel", "opening_pcb_breadth",
        "opening_index_ret", "opening_overseas_votes_present",
    ]], on="day", how="left")
    z.loc[z.bar_no < OPEN_BARS - 1, "opening_regime"] = "PENDING"
    return z, daily


def _metrics(a: pd.DataFrame):
    if a.empty:
        return {
            "n": 0, "fixed_1_5": None, "atr_0_75": None,
            "directional": None, "median_mfe": None, "median_mae": None,
            "median_ratio": None,
        }
    z = a.copy()
    valid = z.atr_pct.notna() & np.isfinite(z.atr_pct) & (z.atr_pct > 0)
    za = z.loc[valid].copy()
    denom = za.mae_30m.where(za.mae_30m > 1e-9, np.nan)
    ratio = za.mfe_30m / denom
    return {
        "n": int(len(z)),
        "fixed_1_5": float((z.future_30m >= .015).mean()),
        "atr_0_75": float((za.mfe_30m >= .75 * za.atr_pct).mean()) if len(za) else None,
        "directional": float((z.mfe_30m > z.mae_30m).mean()),
        "median_mfe": float(z.mfe_30m.median()),
        "median_mae": float(z.mae_30m.median()),
        "median_ratio": float(ratio.median()) if ratio.notna().any() else None,
    }


def _opening_accuracy(daily: pd.DataFrame):
    a = daily[daily.opening_ready].copy()
    if a.empty:
        return {}
    out = {
        "days": int(len(a)),
        "exact_type_accuracy": float((a.opening_regime == a.truth_day_type).mean()),
        "directional_accuracy_ex_mixed_prediction": None,
        "regime_counts": a.opening_regime.value_counts().to_dict(),
        "truth_counts": a.truth_day_type.value_counts().to_dict(),
        "overseas_vote_days": int((a.opening_overseas_votes_present > 0).sum()),
    }
    d = a[a.opening_regime.isin(["UP", "DOWN"])]
    if len(d):
        out["directional_accuracy_ex_mixed_prediction"] = float(
            (d.opening_regime == d.truth_day_type).mean()
        )
    return out


def _daily_type_stats(daily: pd.DataFrame):
    out = {}
    for regime in ["UP", "DOWN", "MIXED"]:
        a = daily[daily.opening_regime == regime]
        if a.empty:
            out[regime] = {"days": 0}
            continue
        out[regime] = {
            "days": int(len(a)),
            "median_final_ret": float(a.final_ret.median()),
            "up_truth_rate": float((a.truth_day_type == "UP").mean()),
            "down_truth_rate": float((a.truth_day_type == "DOWN").mean()),
            "mixed_truth_rate": float((a.truth_day_type == "MIXED").mean()),
            "high_after_10_rate": float((a.high_bar_no >= OPEN_BARS).mean()),
            "low_after_10_rate": float((a.low_bar_no >= OPEN_BARS).mean()),
        }
    return out


def _attach_action_features(x, actionable):
    if actionable.empty:
        return actionable
    z = x.copy()
    rng = (z.high - z.low).replace(0, np.nan)
    z["close_pos_bar"] = (z.close - z.low) / rng
    z["bar_red"] = z.close < z.open
    z["pcb_weak"] = z.pcb_rel_chg3 < 0
    z["breadth_weak"] = z.pcb_breadth_chg3 < 0
    z["below_vwap"] = z.dist_vwap < 0
    z["lower_half"] = z.close_pos_bar <= .50
    z["ret3_decel"] = z.ret3_accel < 0
    cols = [
        "ts", "opening_regime", "opening_up_votes", "opening_down_votes",
        "bar_red", "pcb_weak", "breadth_weak", "below_vwap",
        "lower_half", "ret3_decel",
    ]
    return actionable.merge(z[cols], on="ts", how="left")


def _high_policy_table(high: pd.DataFrame):
    if high.empty:
        return {}
    two_of_four = (
        high[["pcb_weak", "bar_red", "lower_half", "breadth_weak"]]
        .fillna(False).astype(int).sum(axis=1) >= 2
    )
    evidence_any = (
        high[["pcb_weak", "bar_red", "lower_half", "breadth_weak"]]
        .fillna(False).any(axis=1)
    )
    masks = {
        "V17_ALL": pd.Series(True, index=high.index),
        "ALL_REQUIRE_PCB_WEAK_OR_RED": high.pcb_weak.fillna(False) | high.bar_red.fillna(False),
        "ALL_REQUIRE_2_OF_4": two_of_four,
        "UP_ONLY_REQUIRE_ANY_EVIDENCE": (high.opening_regime != "UP") | evidence_any,
        "UP_ONLY_REQUIRE_PCB_WEAK_OR_RED": (
            (high.opening_regime != "UP") |
            high.pcb_weak.fillna(False) |
            high.bar_red.fillna(False)
        ),
        "UP_ONLY_REQUIRE_2_OF_4": (high.opening_regime != "UP") | two_of_four,
    }
    base_n = len(high)
    out = {}
    for name, mask in masks.items():
        a = high[pd.Series(mask, index=high.index).fillna(False)]
        m = _metrics(a)
        m["coverage_vs_v17"] = float(len(a) / base_n) if base_n else None
        out[name] = m
    return out


def main():
    x = e14.build_scored_frame()
    x, daily = add_opening_regime(x)
    ev, actionable = e17.build_v17(x)
    actionable = _attach_action_features(x, actionable)

    high = actionable[actionable.side == "HIGH"].copy()
    low = actionable[actionable.side == "LOW"].copy()

    by_regime = {"HIGH": {}, "LOW": {}}
    for regime in ["PENDING", "UP", "DOWN", "MIXED"]:
        by_regime["HIGH"][regime] = _metrics(high[high.opening_regime == regime])
        by_regime["LOW"][regime] = _metrics(low[low.opening_regime == regime])

    report = {
        "version": "opening-regime-diagnostics-v18",
        "opening_definition": {
            "bars": OPEN_BARS,
            "available_from": "sixth completed 5-minute bar (normally 10:00)",
            "labels": ["UP", "DOWN", "MIXED"],
            "candidate_generation_changed": False,
            "low_policy_changed": False,
            "no_future_leakage": True,
        },
        "trading_days": int(x.ts.dt.date.nunique()),
        "opening_accuracy": _opening_accuracy(daily),
        "daily_regime_stats": _daily_type_stats(daily),
        "v17_actions_by_opening_regime": by_regime,
        "high_quick_confirm_policy_candidates": _high_policy_table(high),
        "note": (
            "Policy candidates only filter action after V1.7 has already produced a HIGH quick structure "
            "confirmation. WATCH/candidate generation is untouched. Use development periods to select a "
            "rule; evaluate the frozen rule on an untouched holdout."
        ),
    }

    daily.to_csv(RESULTS / "opening_regime_daily_v18.csv", index=False)
    actionable.to_csv(RESULTS / "opening_regime_actions_v18.csv", index=False)
    (RESULTS / "opening_regime_diagnostics_v18.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("OPENING REGIME DIAGNOSTICS V1.8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
