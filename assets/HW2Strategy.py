from backtesting import Strategy
import pandas as pd
import numpy as np
from assets.DataProvider import DataProvider


class HW2Strategy_SMA_RSI(Strategy):
    # Define parameters
    sma_short = 10
    sma_long = 20
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    
    # Risk management parameters
    atr_period = 14  # ATR period for volatility calculation
    risk_per_trade = 0.02  # Risk 2% per trade
    trailing_stop_atr = 2.0  # Trailing stop distance in ATR units
    initial_stop_atr = 3.0  # Initial stop loss distance in ATR units
    
    def init(self):
        # Calculate indicators for visualization
        close_series = pd.Series(self.data.Close)
        self.sma_short_line = self.I(lambda x: x.rolling(window=self.sma_short).mean(), close_series)
        self.sma_long_line = self.I(lambda x: x.rolling(window=self.sma_long).mean(), close_series)
        
        # Calculate RSI
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        self.rsi = self.I(lambda x: 100 - (100 / (1 + rs)), close_series)
        
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
        # Check if we have enough data
        if pd.isna(self.sma_long_line[-1]) or pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1]):
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
        rsi_oversold = self.rsi[-1] < self.rsi_lower
        strong_uptrend = trend_strength > 0.015
        
        # Sell signal conditions
        sma_cross_down = (self.sma_short_line[-1] < self.sma_long_line[-1] and
                         self.sma_short_line[-2] >= self.sma_long_line[-2])
        rsi_overbought = self.rsi[-1] > self.rsi_upper
        strong_downtrend = trend_strength < -0.015
        
        # Execute signals
        if not self.position:
            if (sma_cross_up and strong_uptrend) or (rsi_oversold and not strong_downtrend):
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
                if sma_cross_down or rsi_overbought or strong_downtrend:
                    self.position.close()
                    self.trailing_stop = None


class HW2Strategy_SMA(Strategy):
    # Define parameters
    sma_short = 10
    sma_long = 20
    
    # Risk management parameters
    atr_period = 14  # ATR period for volatility calculation
    risk_per_trade = 0.02  # Risk 2% per trade
    trailing_stop_atr = 2.0  # Trailing stop distance in ATR units
    initial_stop_atr = 3.0  # Initial stop loss distance in ATR units
    
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


''' 
Risk management
1. Position Sizing (Risk Management):
Risk per trade: 2% of equity
Position size calculated using ATR (Average True Range)
Formula: position_size = (equity * risk_per_trade) / stop_distance
This ensures consistent risk across trades regardless of volatility
2. Stop Loss System:
Initial stop: 3 ATR units below entry price
Trailing stop: 2 ATR units below price
Trailing stop only moves up, never down
Position automatically closes if price hits stop loss
3. ATR Calculation:
Period: 14 days
Uses True Range: max(high-low, |high-prev_close|, |low-prev_close|)
Helps measure volatility for both position sizing and stops



'''