from assets.DataProvider import DataProvider
from assets.HW2Strategy import HW2Strategy
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataResolution, DataPeriod
import datetime as dt

def main():
    # Initialize data provider with BTC/USDT data
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_02
    )
    
    # Load the data
    data_provider.data_load()
    
    # Test both strategies
    
    # 1. Original backtesting strategy
    btc_data = data_provider.data['BTC/USDT']
    backtester = HW2Backtest(btc_data)
    best_params = backtester.get_best_strategy(HW2Strategy)
    validation_stats = backtester.bactest_strategy(
        backtester.validation_data,
        HW2Strategy,
        best_params
    )
    
    # 2. Test our new StrategyHW2
    from assets.StrategyHW2 import StrategyHW2
    strategy = StrategyHW2(data_provider)
    signals = strategy.run()
    
    # Create performance dashboard
    strategies_results = {
        'Best Strategy': validation_stats
    }
    backtester.create_performance_dashboard(strategies_results)
    
    print("\nBacktesting Strategy Results:")
    print(validation_stats)
    
    print("\nStrategyHW2 Signals:")
    if len(signals) > 0:
        print(f"Total signals generated: {len(signals)}")
        print("\nFirst 5 signals:")
        print(signals.head())
    else:
        print("No signals generated")

if __name__ == "__main__":
    main()
