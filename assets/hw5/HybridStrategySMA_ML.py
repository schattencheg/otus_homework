#%pip install backtesting
#%pip install nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from assets.FeaturesGenerator import FeaturesGenerator
from backtesting import Backtest, Strategy


class HybridStrategySMA_ML(Strategy):
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
        
        #if self.predicted == 1:
        #    print('Predicted to enter!')
        # Execute signals
        if not self.position:
            if (self.model is None and (sma_cross_up and strong_uptrend) or (not strong_downtrend)) or \
                (self.model is not None and self.predicted == 1):
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
