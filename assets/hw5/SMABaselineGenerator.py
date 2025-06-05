#%pip install backtesting
#%pip install nbformat
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataPeriod, DataResolution
from assets.hw5.HybridStrategySMA_ML import HybridStrategySMA_ML
from backtesting import Backtest, Strategy


class SMABaselineGenerator:
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
        # Resulting values
        self.best_params = None
        self.best_results = None
        self.best_trades = None

    def find_optimal_parameters(self, strategy_name: str, strategy_class: Strategy, param_grid):
        print(f'Running {strategy_name} strategy validation...')
        best_params, best_result = self.backtester.get_best_strategy(strategy_class, param_grid)
        return best_params, best_result

    def get_trades(self, name, strategy_class, params):
        results = self.backtester.backtest_strategy(self.data, strategy_class, params)
        trades = results._trades
        return results, trades

    def evaluate_strategies(self):
        self.best_params, self.best_results = self.find_optimal_parameters('SMA', HybridStrategySMA_ML, self.param_grid_sma)
        self.best_trades = self.best_results._trades
        #self.results, self.trades_sma = self.get_trades('SMA', HybridStrategySMA_ML, self.params_sma)
        return self.best_results, self.best_params, self.best_trades

    def evaluate_strategy(self, params):
        results, trades = self.get_trades('SMA', HybridStrategySMA_ML, params)
        return results, trades
    
    @property
    def data(self):
        return self.data_provider.data[self.ticker]

