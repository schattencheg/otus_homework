#%pip install backtesting
#%pip install nbformat
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from assets.DataProvider import DataProvider
from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.HW2Backtest import HW2Backtest
from assets.HW2Strategy import HW2Strategy_SMA, HW2Strategy_SMA_RSI
from assets.enums import DataPeriod, DataResolution
from backtesting import Strategy

class Strategy_HW2:
    def __init__(self, ticker: str = 'BTC/USDT', timeframe: str = '1h', 
                        data_provider: DataProvider = None):
        self.ticker = ticker
        self.timeframe = timeframe
        # Initialize data provider with BTC/USDT data
        if data_provider is None:
            data_provider = DataProvider(
                tickers=['BTC/USDT'],
                resolution=DataResolution.DAY_01,
                period=DataPeriod.YEAR_05)
        self.data_provider = data_provider
        # Load the data
        if ticker not in self.data_provider.data:
            if not self.data_provider.data_load():
                self.data_provider.data_request()
                self.data_provider.data_save()
            self.data_provider.clean_data()
        self.backtester: HW2Backtest = HW2Backtest(self.data)
        # Define parameter grids for each strategy
        self.param_grid_sma = {
            'sma_short': [5, 8],
            'sma_long': [15, 20]}

        self.param_grid_sma_rsi = {
            'sma_short': [5, 8],
            'sma_long': [15, 20],
            'rsi_period': [10, 14],
            'rsi_upper': [65, 70],
            'rsi_lower': [20, 25]}

    def find_optimal_parameters(self, strategy_name: str, strategy_class: Strategy, param_grid):
        print(f'Running {strategy_name} strategy validation...')
        best_params = self.backtester.get_best_strategy(strategy_class, param_grid)
        return best_params[0]

    def get_trades(self, name, strategy_class, params):
        trades = self.backtester.backtest_strategy(self.data, strategy_class, params)._trades
        return trades

    def evaluate_strategies(self):
        self.params_sma = self.find_optimal_parameters('SMA', HW2Strategy_SMA, self.param_grid_sma)
        self.trades_sma = self.get_trades('SMA', HW2Strategy_SMA, self.params_sma)
        #
        self.params_sma_rsi = self.find_optimal_parameters('SMA_RSI', HW2Strategy_SMA_RSI, self.param_grid_sma_rsi)
        self.trades_sma_rsi = self.get_trades('SMA_RSI', HW2Strategy_SMA_RSI, self.params_sma_rsi)

    @property
    def data(self):
        return self.data_provider.data[self.ticker]

    def prepare_targets(self):
        # 1. Generate trades, using homework 2
        self.evaluate_strategies()
        trades_sma = strategy_hw2.trades_sma
        params_sma = strategy_hw2.params_sma
        # 2. Extract data for ML
        trades_dates = trades_sma['EntryTime'].values
        trades_pnl = trades_sma['PnL'].values
        trades_return_pct = trades_sma['ReturnPct'].values
        y = [1 if x > threshold else 0 for x in trades_return_pct]
        # 3. Train ML model on extracted data
        # Add PnL and Returns from strategy run
        data_extended = pd.concat([data, trades_sma[['PnL', 'ReturnPct']]], axis=1)
