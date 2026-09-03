import backtest_v13 as v13
from backtest_v14_cloud import load_a_cloud
import watch_predictor_diagnostics as diag


if __name__ == "__main__":
    v13.load_a = load_a_cloud
    diag.main()
