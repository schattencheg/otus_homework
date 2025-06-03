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


class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x


def main(ticker='BTC/USDT', timeframe='1h'):
    ticker = ticker
    timeframe = timeframe
    model = None
    scaler = StandardScaler()
    data_provider = DataProvider(tickers=[ticker], skip_dashboard=True)
    metrics = {}
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    # 0. Initial parameters
    ticker = 'BTC/USDT'
    timeframe = '1h'
    threshold = 0.2
    # Load the data
    data_provider = DataProvider(tickers=[ticker], skip_dashboard=True)
    if not data_provider.data_load():
        data_provider.data_request()
        data_provider.data_save()
    data_provider.clean_data()
    data = data_provider.data[ticker]
    # 1. Generate trades, using homework 2
    strategy_hw2 = Strategy_HW2(ticker, timeframe, data_provider)
    strategy_hw2.evaluate_strategies()
    trades_sma = strategy_hw2.trades_sma
    params_sma = strategy_hw2.params_sma
    trades_sma_rsi = strategy_hw2.trades_sma_rsi
    params_sma_rsi = strategy_hw2.params_sma_rsi
    # 2. Extract data for ML
    trades_dates = trades_sma['EntryTime'].values
    trades_pnl = trades_sma['PnL'].values
    trades_return_pct = trades_sma['ReturnPct'].values
    y = [1 if x > threshold else 0 for x in trades_return_pct]
    # 3. Train ML model on extracted data
    # Add PnL and Returns from strategy run
    data_extended = pd.concat([data, trades_sma[['PnL', 'ReturnPct']]], axis=1)

    features_generator = FeaturesGenerator()
    X = data.loc[trades_dates].values
    X, feature_columns = features_generator.prepare_features(X)
    main()
