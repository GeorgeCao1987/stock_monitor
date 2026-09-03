from pathlib import Path
import json
import numpy as np
import pandas as pd

import event_engine_v14 as e14
import rolling_extreme_probability_v23 as v23

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# V2.5 is a NEW rule selected only from 2026 Jan-Jun development.
# 2025 Sep-Oct was already consumed by V2.4 and is forbidden for V2.5 tuning/validation.
POS_T_RET_FROM_OPEN_MAX = -0.030
POS_T_DIST_VWAP_MAX = -0.005
POS_T_START_BAR = 3
POS_T_END_BAR = 23
MIN_BOUNCE = 0.008


def episode_starts(z: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    q = z.copy().sort_values(["day", "ts"])
    q["_cond"] = pd.Series(condition, index=z.index).reindex(q.index).fillna(False).astype(bool)
    prev = q.groupby("day")["_cond"].shift(1).fillna(False)
    return q[q._cond & (~prev)].drop(columns="_cond")


def build_rows() -> pd.DataFrame:
    x = e14.build_scored_frame()
    z = v23.add_v23_state(x)
    z = z[z.eligible_realtime].copy()
    cond = (
        z.bottom_tradeable_now
        & (z.ret_from_open <= POS_T_RET_FROM_OPEN_MAX)
        & (z.dist_vwap <= POS_T_DIST_VWAP_MAX)
        & z.bar_idx.between(POS_T_START_BAR, POS_T_END_BAR)
    )
    q = episode_starts(z, cond).copy()
    q["bounce_hit"] = q.remaining_upside_from_close >= MIN_BOUNCE
    q["directional_edge"] = q.remaining_upside_from_close > q.remaining_downside_from_close
    q["positive_t_edge_success"] = q.bounce_hit & q.directional_edge
    q["exact_bottom_locked"] = q.future_low_after >= q.cum_low
    return q


def metrics(q: pd.DataFrame) -> dict:
    if q.empty:
        return {"signals": 0, "signal_days": 0}
    ratio = q.remaining_upside_from_close / q.remaining_downside_from_close.replace(0, np.nan)
    return {
        "signals": int(len(q)),
        "signal_days": int(q.day.nunique()),
        "primary_edge_success_rate": float(q.positive_t_edge_success.mean()),
        "bounce_0_8_hit_rate": float(q.bounce_hit.mean()),
        "directional_edge_rate": float(q.directional_edge.mean()),
        "exact_final_bottom_locked_rate": float(q.exact_bottom_locked.mean()),
        "median_future_upside": float(q.remaining_upside_from_close.median()),
        "median_future_downside": float(q.remaining_downside_from_close.median()),
        "median_upside_downside_ratio": float(ratio.replace([np.inf, -np.inf], np.nan).median()) if ratio.notna().any() else None,
        "median_ret_from_open": float(q.ret_from_open.median()),
        "median_dist_vwap": float(q.dist_vwap.median()),
        "time_phases": q.time_phase.value_counts().to_dict(),
    }


def main():
    q = build_rows()
    report = {
        "version": "t-edge-v25",
        "status": "new_frozen_candidate",
        "objective": "Positive-T rebound edge, independent from final-bottom lock probability.",
        "rule": {
            "bottom_tradeable_now": True,
            "ret_from_open_lte": POS_T_RET_FROM_OPEN_MAX,
            "dist_vwap_lte": POS_T_DIST_VWAP_MAX,
            "window": "09:50-11:30",
            "episode_counting": "false-to-true only",
        },
        "primary_win": "future upside >= 0.8% AND future upside > future downside",
        "development_record_before_holdout": {
            "periods": "2026 Jan-Feb / Mar-Apr / May-Jun",
            "success_rates": [0.7692307692, 0.7333333333, 0.7222222222],
            "signals": [13, 15, 18],
            "pooled_rate": 0.7391304348,
            "pooled_signals": 46,
        },
        "forbidden_holdout": "2025 Sep-Oct already viewed for V2.4; never use it to validate/tune V2.5.",
        "future_leakage": False,
        "metrics": metrics(q),
    }
    q.to_csv(RESULTS / "t_edge_v25_signals.csv", index=False)
    (RESULTS / "t_edge_v25.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("T EDGE V2.5")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
