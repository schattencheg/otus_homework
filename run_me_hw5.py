#%pip install backtesting
#%pip install nbformat
import os
import traceback
from matplotlib import pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
import torch
import numpy as np
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataPeriod, DataResolution
from backtesting import Backtest, Strategy


class HW2Strategy_SMA(Strategy):
    # Define parameters
    sma_short = 10
    sma_long = 20
    model = None
    predicted = None
    window_size = 30       # Default, overridden if passed in bt.run()
    features_generator = FeaturesGenerator()
    
    # Risk management parameters
    atr_period = 14  # ATR period for volatility calculation
    risk_per_trade = 0.02  # Risk 2% per trade
    trailing_stop_atr = 2.0  # Trailing stop distance in ATR units
    initial_stop_atr = 3.0  # Initial stop loss distance in ATR units
    
    def _get_feature_window(self):
        feature_calc_buffer = 60 
        required_ohlc_length = self.window_size + feature_calc_buffer

        if len(self.data.df) < required_ohlc_length:
            return None

        ohlc_for_features = self.data.df.iloc[-required_ohlc_length:].copy()
        ohlc_for_features = self.data.df.copy()
        all_features_df, _ = self.features_generator.prepare_features(ohlc_for_features)

        if len(all_features_df) < self.window_size:
            return None
        
        feature_window_np = all_features_df.iloc[-self.window_size:].values
        
        if feature_window_np.shape[0] != self.window_size:
            return None
            
        return feature_window_np

    def predict_signal(self) -> int:
        if not self.model:
            return -1 # No model, so hold/no signal

        feature_window = self._get_feature_window()
        if feature_window is None:
            return -1 # Cannot get features, hold

        if not isinstance(feature_window, np.ndarray):
            return -1 # Invalid features, hold

        try:
            features_tensor = torch.tensor(feature_window, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            # print(f"DEBUG STRATEGY: Error converting feature_window to tensor: {e}")
            return -1 # Error, hold

        self.model.eval()
        with torch.no_grad():
            try:
                raw_prediction = self.model(features_tensor)
            except Exception as e:
                # print(f"DEBUG STRATEGY: Error during model prediction: {e}")
                return -1 # Error, hold
        
        predicted_class = torch.argmax(raw_prediction, dim=1).item()
        return int(predicted_class) # 0 or 1

    def init(self):
        # Calculate indicators for visualization
        close_series = pd.Series(self.data.Close)
        self.sma_short_line = self.I(lambda x: x.rolling(window=self.sma_short).mean(), close_series)
        self.sma_long_line = self.I(lambda x: x.rolling(window=self.sma_long).mean(), close_series)
        
        # Calculate ATR for position sizing and stops
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        
        tr1 = high - low  # Current high - current low
        tr2 = abs(high - close.shift())  # Current high - previous close
        tr3 = abs(low - close.shift())  # Current low - previous close
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = self.I(lambda x: x.rolling(window=self.atr_period).mean(), tr)
        
        # Initialize stop loss tracking
        self.trailing_stop = None
        self.initial_stop = None
    
    def next(self):
        if self.model is not None:
            features, feature_columns = self.features_generator.prepare_features(self.data.df.copy())
            if len(features) > 0:
                self.predicted = self.predict_signal()
        # Check if we have enough data
        if pd.isna(self.sma_long_line[-1]) or pd.isna(self.atr[-1]):
            return
            
        current_price = self.data.Close[-1]
        
        # Update trailing stop if in position
        if self.position:
            if self.position.is_long:
                new_trailing_stop = current_price - self.trailing_stop_atr * self.atr[-1]
                if new_trailing_stop > self.trailing_stop:
                    self.trailing_stop = new_trailing_stop
                
                # Check if stop loss is hit
                if current_price < self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
                    return
        
        # Calculate trend strength
        trend_strength = (self.sma_short_line[-1] - self.sma_long_line[-1]) / self.sma_long_line[-1]
        
        # Buy signal conditions
        sma_cross_up = (self.sma_short_line[-1] > self.sma_long_line[-1] and
                       self.sma_short_line[-2] <= self.sma_long_line[-2])
        strong_uptrend = trend_strength > 0.015
        
        # Sell signal conditions
        sma_cross_down = (self.sma_short_line[-1] < self.sma_long_line[-1] and
                         self.sma_short_line[-2] >= self.sma_long_line[-2])
        strong_downtrend = trend_strength < -0.015
        
        if self.predicted == 1:
            print('Predicted to enter!')
        # Execute signals
        if not self.position:
            if (sma_cross_up and strong_uptrend) or (not strong_downtrend):
                # Calculate position size based on volatility
                stop_distance = self.initial_stop_atr * self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                # Calculate size in units, rounded to nearest whole number
                position_size = max(1, int(risk_amount / (stop_distance * current_price)))
                
                # Set initial stop loss
                self.trailing_stop = current_price - stop_distance
                
                # Enter position
                self.buy(size=position_size)
        else:
            if self.position.is_long:
                # Exit conditions
                if sma_cross_down or strong_downtrend:
                    self.position.close()
                    self.trailing_stop = None


    def predict(self) -> float:
        data_values = self.data.df.values
        X_np = []
        for i in range(len(data_values) - self.window_size):
            X_np.append(data_values[i:(i + self.window_size)])
        return self.model(np.array(X_np))

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

    def evaluate_strategy(self, params):
        trades = self.get_trades('SMA', HW2Strategy_SMA, params)
        return trades
    
    @property
    def data(self):
        return self.data_provider.data[self.ticker]


class StockCNN(nn.Module):
    def __init__(self, input_channels, window_size, kernel_size_conv1=3, kernel_size_conv2=3):
        super(StockCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=kernel_size_conv1)  # Input: dynamic channels, Output: 16 channels
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size_conv2)  # Input: 16 channels, Output: 32 channels
        self.fc1 = nn.Linear(32 * (window_size - (kernel_size_conv1 - 1) -(kernel_size_conv2 - 1)), 64)  # Flattened size after conv
        self.fc2 = nn.Linear(64, 3)  # Output: 3 classes (Buy, Hold, Sell)

    def forward(self, x):
        # Input x shape: (batch_size, window_size, num_features)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, num_features, window_size)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output from convolution layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Return logits (raw output) for each class


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x


class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        _, h_n = self.gru(x)
        x = self.fc(h_n[-1])
        return x


class StockCNN_LSTM(nn.Module):
    def __init__(self, input_channels):
        super(StockCNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3)  # Input: dynamic channels, Output: 16 channels
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)  # Input: 16 channels, Output: 32 channels

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(64, 3)  # Output: 3 classes (Buy, Hold, Sell)

    def forward(self, x):
        # Input x shape: (batch_size, window_size, num_features)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, num_features, window_size) for Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Reshape for LSTM: [batch, seq_length, features]

        x, _ = self.lstm(x)  # Pass through LSTM
        x = x[:, -1, :]  # Take the last time step

        return self.fc1(x)  # Return logits for each class



def main(ticker='BTC/USDT', timeframe='1h'):
    # 0. Initial parameters
    best_params_sma = {} # Initialize to handle cases where SMA optimization might be skipped
    ticker = ticker
    timeframe = timeframe
    threshold = 0.2
    # 1. Load the data
    data_provider = DataProvider(tickers=[ticker], skip_dashboard=True)
    if ticker not in data_provider.data:
        if not data_provider.data_load():
            data_provider.data_request()
            data_provider.data_save()
        data_provider.clean_data()
    data = data_provider.data[ticker]
    # 2. Add features
    features_generator = FeaturesGenerator()
    data_initial_ohlc = data.copy()  # Store a true copy of original OHLC data
    feature_matrix, feature_names = features_generator.prepare_features(data_initial_ohlc) # Pass original; 'data' is now data_ohlc_aligned_with_features
    num_features = feature_matrix.shape[1]
    print(f"Number of features generated: {num_features} ({len(feature_names)} names)")
    # 3. Generate target
    # 3.1. Generate trades, using homework 2 (only long)
    strategy_hw2 = Strategy_HW2(ticker, timeframe, data_provider)
    strategy_hw2.evaluate_strategies()
    trades_sma = strategy_hw2.trades_sma
    params_sma = strategy_hw2.params_sma
    # 3.2. Extract data for ML
    # Add PnL and Returns from strategy run as additional features
    data_extended = feature_matrix.copy() 
    if trades_sma is None or trades_sma.empty:
        print("WARN: No trades from SMA strategy. PnL and ReturnPct will be zero for y_base generation.")
        data_extended['PnL'] = 0.0
        data_extended['ReturnPct'] = 0.0
    else:
        trades_sma = trades_sma.set_index('EntryTime')
        # Merge PnL and ReturnPct from trades_sma into data_extended
        # Use a left merge to keep all rows from data_extended (which is aligned with features)
        # and fill missing PnL/ReturnPct with 0 for non-trade days or if trades_sma is shorter
        data_extended = data_extended.merge(trades_sma[['PnL', 'ReturnPct']], 
                                            left_index=True, right_index=True, how='left')
        data_extended[['PnL', 'ReturnPct']] = data_extended[['PnL', 'ReturnPct']].fillna(value=0)
    trades_return_pct = data_extended['ReturnPct'].values
    y_base = [1 if x > threshold else 0 for x in trades_return_pct]
    # 4. Dataset generation
    print(f"DEBUG: Shape of feature_matrix before create_sequences: {feature_matrix.shape}")
    print(f"DEBUG: Length of y_base before create_sequences: {len(y_base)}")
    window_size = 30  # Define window_size
    # Ensure feature_matrix.values and y_base are used with the create_sequences function
    X_np, y_np = create_sequences(feature_matrix.values, np.array(y_base), window_size)
    print(f"DEBUG: Shape of X_np after create_sequences: {X_np.shape}")
    print(f"DEBUG: Shape of y_np after create_sequences: {y_np.shape}")
    # Now convert lists to tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)  # Use y_np here
    # DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # ... (rest of the code remains the same)
    models = {'CNN': {  'model': StockCNN(input_channels=num_features, window_size=window_size), 
                        'title': 'CNN model Predictions vs. Actual',
                        'model_trained': None},
              'LSTM': { 'model': StockLSTM(input_size=num_features), 
                        'title': 'LSTM model Predictions vs. Actual',
                        'model_trained': None},
              'GRU': {  'model': StockGRU(input_size=num_features), 
                        'title': 'GRU model Prediction vs. Actual',
                        'model_trained': None},
              'CNN-LSTM': { 'model': StockCNN_LSTM(input_channels=num_features), 
                            'title': 'CNN-LSTM model Prediction vs. Actual',
                            'model_trained': None}}
    
    models = {'CNN': {  'model': StockCNN(input_channels=num_features, window_size=window_size), 
                        'title': 'CNN model Predictions vs. Actual',
                        'model_trained': None}}
    
    # 6. Train models
    for model in models:
        print(f"Training {model} model...")
        model_trained = train_model(models[model]['title'], models[model]['model'], dataloader)
        models[model]['model_trained'] = model_trained
    # 7. Test models
    for model in models:
        test_model(models[model]['title'], models[model]['model_trained'], dataloader)
    # 8. Save models
    for model in models:
        if not os.path.exists(f"hw5"):
            os.makedirs(f"hw5")
        torch.save(models[model]['model_trained'].state_dict(), f"hw5/{model}_model_trained.pt")
    # 9. Run models over real strategy
    print("\n--- Running Backtests with Trained Models ---")
    all_backtest_results = {} # Initialize dictionary to store results
    for model_key in models: # Renamed loop variable from 'model' for clarity
        print(f"\n\n\nPreparing backtest for model: {model_key}...")
        # HW2Strategy_SMA (in assets/StrategyCollection.py) will need to be modified 
        # to accept and use model, window_size, feature_names, and scaler parameters.
        # For now, instantiating with original SMA parameters to allow the script to run.
        # This backtest will use SMA logic, NOT the trained model.
        print(f"INFO: HW2Strategy_SMA (defined in this file) needs update for model: {model_key}.")
        print("INFO: Running backtest with default SMA logic for now.")
        # The 'data' DataFrame passed to Backtest is the one after 'features_generator.prepare_features(data)'
        # It must contain 'Open', 'High', 'Low', 'Close', 'Volume' and all feature columns.
        print(f"Running backtest for {model_key} using SMA logic (model not integrated yet).")
        # Ensure 'data' is the DataFrame with features included
        # Pass HW2Strategy_SMA class to Backtest constructor
        bt = Backtest(data.copy(),  # Pass data.copy()
                      HW2Strategy_SMA, 
                      cash=10000000, 
                      commission=.002)
        try:
            # Pass sma_short and sma_long to bt.run()
            stats = bt.run(
                sma_short=params_sma.get('sma_short', HW2Strategy_SMA.sma_short),
                sma_long=params_sma.get('sma_long', HW2Strategy_SMA.sma_long),
                model=models[model_key]['model_trained']
            )
            print(f"--- Backtest Results for {model_key} driven strategy ---")
            print(stats)
            all_backtest_results[model_key] = {'stats': stats}
            plot_filename = f"backtest_plot_{model_key}.html"
            try:
                bt.plot(filename=plot_filename, open_browser=False)
                all_backtest_results[model_key]['plot_file'] = plot_filename
                print(f"Backtest plot for {model_key} saved to {plot_filename}")
            except Exception as e:
                print(f"ERROR: Could not generate plot for {model_key}: {e}")
                all_backtest_results[model_key]['plot_file'] = None
        except Exception as e:
            print(f"ERROR during backtest for {model_key}: {e}")
            print("This might be due to HW2Strategy_SMA not being fully adapted for model-based trading,")
            print("or issues with data slicing/feature access within the strategy's next() method.")
            traceback.print_exc() # Print detailed traceback
    print("\n\n--- Summary of All Backtest Results ---")
    for model_key_summary, results_summary in all_backtest_results.items(): # Renamed loop variables for clarity
        print(f"\n--- Results for {model_key_summary} ---")
        print(results_summary['stats'])
        if 'plot_file' in results_summary and results_summary['plot_file']:
            print(f"Plot saved to: {results_summary['plot_file']}")
        else:
            print("Plot generation failed or was not attempted.")

def train_model(title, model, dataloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    return model
    #
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())

    # Plot actual vs predicted labels
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[-30:], label='Actual', color='blue', linestyle='--', alpha=0.7)
    plt.plot(all_predictions[-30:], label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Signal (0=Sell, 1=Hold, 2=Buy)")
    plt.legend()
    plt.show()

def test_model(title, model, dataloader):
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"{title} Accuracy: {accuracy:.4f}")
    return accuracy
    # Plot actual vs predicted labels
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[-30:], label='Actual', color='blue', linestyle='--', alpha=0.7)
    plt.plot(all_predictions[-30:], label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Signal (0=Sell, 1=Hold, 2=Buy)")
    plt.legend()
    plt.show()
    
    return accuracy

def create_sequences(data_values, target_values, window_size):
    X, y = [], []
    for i in range(len(data_values) - window_size):
        X.append(data_values[i:(i + window_size)])
        y.append(target_values[i + window_size])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    ticker = 'BTC/USDT'
    timeframe = '1h'
    main(ticker, timeframe)
