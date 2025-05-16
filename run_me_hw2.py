#%pip install backtesting
#%pip install nbformat
from assets.DataProvider import DataProvider
from assets.HW2Strategy import HW2Strategy_SMA_RSI, HW2Strategy_SMA
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataResolution, DataPeriod
from backtesting import Strategy
import nbformat
import datetime as dt

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
data_provider.clean_data()

btc_data = data_provider.data['BTC/USDT']
backtester: HW2Backtest = HW2Backtest(btc_data)

# Define parameter grids for each strategy

param_grid_sma = {
    'sma_short': [5, 8],
    'sma_long': [15, 20]
}

param_grid_sma_rsi = {
    'sma_short': [5, 8],
    'sma_long': [15, 20],
    'rsi_period': [10, 14],
    'rsi_upper': [65, 70],
    'rsi_lower': [20, 25]
}

def test_strategy(strategy_name: str, strategy_class: Strategy, param_grid, backtester: HW2Backtest):
    print(f'Running {strategy_name} strategy validation...')
    best_params = backtester.get_best_strategy(strategy_class, param_grid)
    validation_stats = backtester.backtest_strategy(
        backtester.validation_data,
        strategy_class,
        best_params[0]
    )
    return validation_stats
    


# Strategy SMA
strategy_name = 'SMA'
strategies_results_sma = test_strategy(strategy_name, HW2Strategy_SMA, param_grid_sma, backtester)
strategies_results = {f'{strategy_name}': strategies_results_sma}


# Strategy SMA_RSI
strategy_name = 'SMA_RSI'
strategies_results_sma_rsi = test_strategy(strategy_name, HW2Strategy_SMA_RSI, param_grid_sma_rsi, backtester)
strategies_results[strategy_name] = strategies_results_sma_rsi


# Create performance dashboard
backtester.create_performance_dashboard(strategies_results)
