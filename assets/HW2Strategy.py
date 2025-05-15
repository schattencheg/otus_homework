from backtesting import Strategy
import pandas as pd
import numpy as np
from .DataProvider import DataProvider
from .StrategyHW2 import StrategyHW2

class HW2Strategy(Strategy):
    # Define parameters
    sma_short = 10
    sma_long = 20
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    
    def init(self):
        # Create a DataProvider with our data
        df = pd.DataFrame({
            'Open': self.data.Open,
            'High': self.data.High,
            'Low': self.data.Low,
            'Close': self.data.Close,
            'Volume': self.data.Volume
        }, index=self.data.index)
        
        data_provider = DataProvider()
        data_provider.data = {'BTC/USDT': df}
        
        # Create StrategyHW2 instance
        self.strategy = StrategyHW2(data_provider)
        
        # Calculate indicators for visualization
        close_series = pd.Series(self.data.Close)
        self.sma_short_line = self.I(lambda x: x.rolling(window=self.sma_short).mean(), close_series)
        self.sma_long_line = self.I(lambda x: x.rolling(window=self.sma_long).mean(), close_series)
        
        # Get signals from StrategyHW2
        signals_df = self.strategy.run()
        
        # Store signals for use in next()
        self.signals = signals_df.set_index('timestamp')
        print(f"Loaded {len(self.signals)} signals from StrategyHW2")

    def next(self):
        # Get current date
        current_date = self.data.index[-1]
        
        # Check if we have a signal for this date
        if current_date in self.signals.index:
            signal = self.signals.loc[current_date]
            
            # Execute the signal with size 1.0 (use all available cash)
            if signal['action'] == 'BUY' and not self.position:
                self.buy(size=1.0)
            elif signal['action'] == 'SELL' and self.position:
                self.sell(size=1.0)
