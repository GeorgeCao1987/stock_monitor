import backtest_v13 as v13
from backtest_v14_cloud import load_a_cloud
import stage_extreme_replay_v19 as replay


if __name__ == "__main__":
    v13.load_a = load_a_cloud
    replay.main()
