from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import rolling_extreme_probability_v23 as v23

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# Frozen from 2026 Jan-Jun development before the 2025 Sep-Oct holdout is read.
POS_T_RET_FROM_OPEN_MAX = -0.020
POS_T_DIST_VWAP_MAX = -0.010
POS_T_START_BAR = 3      # ~09:50 with current 5m bar convention
POS_T_END_BAR = 23       # ~11:30
POS_T_MIN_BOUNCE = 0.008


def episode_starts(z: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    q = z.copy().sort_values(["day", "ts"])
    q["_cond"] = pd.Series(condition, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False)
    return q[q._cond & (~prev)].drop(columns="_cond")


def build_rows() -> pd.DataFrame:
    x = e14.build_scored_frame()
    z = v23.add_v23_state(x)
    z = z[z.eligible_realtime].copy()

    # IMPORTANT: This is a T-edge condition, NOT a claim that the final daily
    # bottom is already locked.  The execution guard comes from V2.3 and allows
    # a wider post-confirmation distance on the LOW side than on the HIGH side.
    condition = (
        z.bottom_tradeable_now
        & (z.ret_from_open <= POS_T_RET_FROM_OPEN_MAX)
        & (z.dist_vwap <= POS_T_DIST_VWAP_MAX)
        & z.bar_idx.between(POS_T_START_BAR, POS_T_END_BAR)
    )
    q = episode_starts(z, condition).copy()

    # Scoring-only future labels.
    q["positive_t_bounce_0_8_hit"] = q.remaining_upside_from_close >= POS_T_MIN_BOUNCE
    q["positive_t_directional_edge"] = q.remaining_upside_from_close > q.remaining_downside_from_close
    q["positive_t_edge_success"] = q.positive_t_bounce_0_8_hit & q.positive_t_directional_edge
    q["exact_bottom_locked"] = q.future_low_after >= q.cum_low
    return q


def metrics(q: pd.DataFrame) -> dict:
    if q.empty:
        return {"signals": 0, "signal_days": 0}
    ratio = q.remaining_upside_from_close / q.remaining_downside_from_close.replace(0, np.nan)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "primary_positive_t_edge_success_rate": float(q.positive_t_edge_success.mean()),
        "bounce_0_8_hit_rate": float(q.positive_t_bounce_0_8_hit.mean()),
        "directional_edge_rate": float(q.positive_t_directional_edge.mean()),
        "exact_final_bottom_locked_rate": float(q.exact_bottom_locked.mean()),
        "median_future_upside": float(q.remaining_upside_from_close.median()),
        "median_future_downside": float(q.remaining_downside_from_close.median()),
        "median_upside_downside_ratio": float(ratio.replace([np.inf, -np.inf], np.nan).median()) if ratio.notna().any() else None,
        "median_ret_from_open_at_signal": float(q.ret_from_open.median()),
        "median_dist_vwap_at_signal": float(q.dist_vwap.median()),
        "time_phases": q.time_phase.value_counts().to_dict(),
    }


def main():
    q = build_rows()
    report = {
        "version": "t-edge-v24",
        "status": "frozen_positive_t_candidate_for_untouched_holdout",
        "objective": "Separate T-trade edge from final-extreme lock probability.",
        "positive_t_rule": {
            "bottom_tradeable_now": True,
            "ret_from_open_lte": POS_T_RET_FROM_OPEN_MAX,
            "dist_vwap_lte": POS_T_DIST_VWAP_MAX,
            "decision_window": "09:50-11:30 (bar_idx 3..23)",
            "episode_counting": "false-to-true transition only",
        },
        "primary_win_definition": (
            "future upside from signal close >= 0.8% AND future upside > future downside; "
            "future values are scoring-only"
        ),
        "important_interpretation": (
            "This signal estimates positive-T rebound edge. It does NOT require or imply that the final daily bottom is locked."
        ),
        "future_leakage": False,
        "metrics": metrics(q),
    }
    q.to_csv(RESULTS / "t_edge_v24_signals.csv", index=False)
    (RESULTS / "t_edge_v24.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("T EDGE V2.4")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
