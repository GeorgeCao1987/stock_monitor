import backtest_v13 as v13
from backtest_v14_cloud import load_a_cloud
import external_event_incremental_v20 as diag


if __name__ == "__main__":
    v13.load_a = load_a_cloud
    diag.main()
