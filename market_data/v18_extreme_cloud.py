import backtest_v13 as v13
from backtest_v14_cloud import load_a_cloud
import opening_extreme_forecast_v18 as opening
import extreme_capture_diagnostics_v18 as capture


if __name__ == "__main__":
    v13.load_a = load_a_cloud
    opening.main()
    capture.main()
