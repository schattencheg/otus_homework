from assets.DataProvider import DataProvider
from assets.HW2Strategy import HW2Strategy_SMA_RSI, HW2Strategy_SMA
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataResolution, DataPeriod
from backtesting import Strategy
import datetime as dt

def main():
    # Initialize data provider with BTC/USDT data
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_05
    )
    
    # Load the data
    if not data_provider.data_load():
        data_provider.data_request()
        data_provider.data_save()
    
    # Test both strategies
    
    # 1. Original backtesting strategy
    btc_data = data_provider.data['BTC/USDT']
    backtester: HW2Backtest = HW2Backtest(btc_data)

    # Define parameter ranges for grid search
    param_grid = {
        'sma_short': [5, 8, 10],
        'sma_long': [15, 20, 25],
        'rsi_period': [10, 14],
        'rsi_upper': [65, 70, 80],
        'rsi_lower': [20, 25, 35]
    }
    param_grid = {
        'sma_short': [5, 8, 10, 12, 15],
        'sma_long': [15, 20, 25, 30],
        'rsi_period': [10, 14, 21],
        'rsi_upper': [65, 70, 75, 80],
        'rsi_lower': [20, 25, 30, 35]
    }

    # Strategy SMA
    strategy_name = 'SMA'
    strategies_results_sma = test_strategy(strategy_name, HW2Strategy_SMA, param_grid, backtester)
    strategies_results = {f'{strategy_name}': strategies_results_sma}

    # Strategy SMA_RSI
    strategy_name = 'SMA_RSI'
    strategies_results_sma_rsi = test_strategy(strategy_name, HW2Strategy_SMA_RSI, param_grid, backtester)
    strategies_results[strategy_name] = strategies_results_sma_rsi

    # Create performance dashboard
    backtester.create_performance_dashboard(strategies_results)
    stop_here = True


def test_strategy(strategy_name: str, strategy_class: Strategy, param_grid, backtester: HW2Backtest):
    print(f'Running {strategy_name} strategy validation...')
    best_params = backtester.get_best_strategy(strategy_class, param_grid)
    validation_stats = backtester.backtest_strategy(
        backtester.validation_data,
        strategy_class,
        best_params[0]
    )
    return validation_stats
    

if __name__ == "__main__":
    main()
