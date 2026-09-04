import json
from pathlib import Path

import joblib

import backtest_v13 as v13
import extreme_running_lock_walkforward_v37 as v37
from backtest_v14_cloud import load_a_cloud

BASE = Path(__file__).resolve().parent
CACHE = BASE / "live_cache"
CACHE.mkdir(exist_ok=True)


def main():
    v13.load_a = load_a_cloud
    raw = v37.build_frame()
    train = raw[(raw.day >= v37.BULL_START) & v37.eligible(raw)].copy()
    if train.day.nunique() < 60:
        raise RuntimeError(f"训练样本不足: {train.day.nunique()} days")

    bundle = {
        "top_model": v37.fit_model(train, "top_locked_running"),
        "bottom_model": v37.fit_model(train, "bottom_locked_running"),
        "train_days": int(train.day.nunique()),
        "train_start": str(min(train.day)),
        "train_end": str(max(train.day)),
        "version": "V3.7",
    }
    joblib.dump(bundle, CACHE / "model_bundle.joblib")
    (CACHE / "model_meta.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if not k.endswith("_model")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("LIVE_MODEL_BUNDLE_READY", bundle["train_days"], bundle["train_start"], bundle["train_end"], flush=True)


if __name__ == "__main__":
    main()
