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

    def next(self):
        # Check if we have enough data
        if pd.isna(self.sma_long_line[-1]) or pd.isna(self.rsi[-1]):
            return
            
        # Calculate trend strength
        trend_strength = (self.sma_short_line[-1] - self.sma_long_line[-1]) / self.sma_long_line[-1]
        
        # Buy signal conditions
        sma_cross_up = (self.sma_short_line[-1] > self.sma_long_line[-1] and
                       self.sma_short_line[-2] <= self.sma_long_line[-2])
        rsi_oversold = self.rsi[-1] < self.rsi_lower
        strong_uptrend = trend_strength > 0.015  # 1.5% minimum trend strength
        
        # Sell signal conditions
        sma_cross_down = (self.sma_short_line[-1] < self.sma_long_line[-1] and
                         self.sma_short_line[-2] >= self.sma_long_line[-2])
        rsi_overbought = self.rsi[-1] > self.rsi_upper
        strong_downtrend = trend_strength < -0.015  # -1.5% minimum trend strength
        
        # Execute signals
        if not self.position:
            # Buy when price is rising and trend is strong
            if (sma_cross_up and strong_uptrend) or (rsi_oversold and not strong_downtrend):
                self.buy(size=1.0)
            #elif (sma_cross_down and strong_downtrend) or (rsi_overbought and not strong_uptrend):
            #    self.sell(size=1.0)
        else:
            if self.position.is_long:
                # Sell when price is falling or trend is weak
                if sma_cross_down or rsi_overbought or strong_downtrend:
                    self.sell(size=1.0)
            else:
                # Buy when price is falling or trend is weak
                if sma_cross_up or rsi_oversold or strong_uptrend:
                    self.buy(size=1.0)


class HW2Strategy_SMA(Strategy):
    # Define parameters
    sma_short = 10
    sma_long = 20
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    
    def init(self):
        # Calculate indicators for visualization
        close_series = pd.Series(self.data.Close)
        self.sma_short_line = self.I(lambda x: x.rolling(window=self.sma_short).mean(), close_series)
        self.sma_long_line = self.I(lambda x: x.rolling(window=self.sma_long).mean(), close_series)
        
    def next(self):
        # Check if we have enough data
        if pd.isna(self.sma_long_line[-1]):
            return
            
        # Calculate trend strength
        trend_strength = (self.sma_short_line[-1] - self.sma_long_line[-1]) / self.sma_long_line[-1]
        
        # Buy signal conditions
        sma_cross_up = (self.sma_short_line[-1] > self.sma_long_line[-1] and
                       self.sma_short_line[-2] <= self.sma_long_line[-2])
        strong_uptrend = trend_strength > 0.015  # 1.5% minimum trend strength
        
        # Sell signal conditions
        sma_cross_down = (self.sma_short_line[-1] < self.sma_long_line[-1] and
                         self.sma_short_line[-2] >= self.sma_long_line[-2])
        strong_downtrend = trend_strength < -0.015  # -1.5% minimum trend strength
        
        # Execute signals
        if not self.position:
            # Buy when price is rising and trend is strong
            if (sma_cross_up and strong_uptrend) or (not strong_downtrend):
                self.buy(size=1.0)
            #elif (sma_cross_down and strong_downtrend) or (not strong_uptrend):
            #    self.sell(size=1.0)
        else:
            if self.position.is_long:
                # Sell when price is falling or trend is weak
                if sma_cross_down or strong_downtrend:
                    self.sell(size=1.0)
            else:
                # Buy when price is falling or trend is weak
                if sma_cross_up or strong_uptrend:
                    self.buy(size=1.0)
