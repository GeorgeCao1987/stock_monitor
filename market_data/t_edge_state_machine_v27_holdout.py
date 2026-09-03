from pathlib import Path
import json

import t_edge_state_machine_v27 as v27

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    z = v27.base_state()
    pw = v27.episode_starts(z, v27.positive_watch_mask(z))
    pe = v27.positive_execute_events(z)
    re = v27.episode_starts(z, v27.reverse_execute_mask(z))

    report = {
        "version": "t-edge-state-machine-v27",
        "evaluation": "untouched_2025_may_jun_holdout_for_frozen_rules",
        "rules_changed_after_holdout_view": False,
        "trading_days": int(z.day.nunique()),
        "POSITIVE_WATCH": v27.metrics(pw, "POSITIVE"),
        "POSITIVE_EXECUTE": v27.metrics(pe, "POSITIVE"),
        "REVERSE_EXECUTE": v27.metrics(re, "REVERSE"),
    }
    if not pe.empty:
        report["POSITIVE_EXECUTE"]["median_wait_bars"] = float(pe.wait_bars.median())

    pw.to_csv(RESULTS / "v27_holdout_positive_watch.csv", index=False)
    pe.to_csv(RESULTS / "v27_holdout_positive_execute.csv", index=False)
    re.to_csv(RESULTS / "v27_holdout_reverse_execute.csv", index=False)
    (RESULTS / "t_edge_state_machine_v27_holdout.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("T EDGE STATE MACHINE V2.7 HOLDOUT")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
