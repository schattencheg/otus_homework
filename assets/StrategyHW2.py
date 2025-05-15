import pandas as pd
import numpy as np
from assets.DataProvider import DataProvider

class StrategyHW2:
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        # Strategy parameters - balanced for volatile markets
        self.sma_short = 8   # Short enough for quick signals, but not too noisy
        self.sma_long = 21   # 21-day average for trend identification
        self.rsi_period = 14  # Standard RSI period
        self.rsi_upper = 70   # Standard overbought level
        self.rsi_lower = 30   # Standard oversold level
        self.in_position = False  # Track if we're in a position
        self.min_trend_strength = 0.015  # 1.5% minimum trend strength
        
    def calculate_rsi(self, data: pd.Series, periods: int) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def run(self):
        signals = []
        
        for ticker in self.data_provider.tickers:
            df = self.data_provider.data[ticker]
            if df is None or len(df) == 0:
                continue
                
            # Filter data to validation period
            validation_start = pd.Timestamp('2019-12-15')
            validation_end = pd.Timestamp('2020-05-12')
            df = df[validation_start:validation_end].copy()
            
            if len(df) == 0:
                print(f"No data in validation period for {ticker}")
                continue
                
            print(f"Validation period: {df.index[0]} to {df.index[-1]}")
                
            # Calculate indicators
            print(f"DataFrame columns: {df.columns}")
            close_series = df['Close']
            sma_short = close_series.rolling(window=self.sma_short).mean()
            sma_long = close_series.rolling(window=self.sma_long).mean()
            rsi = self.calculate_rsi(close_series, self.rsi_period)
            
            # Generate signals
            for i in range(1, len(df)):
                # Only consider data after both SMAs have enough data
                if pd.isna(sma_long.iloc[i]) or pd.isna(rsi.iloc[i]):
                    continue
                
                # Debug values
                if i % 100 == 0:  # Print every 100th data point
                    print(f"Date: {df.index[i]}, SMA Short: {sma_short.iloc[i]:.2f}, SMA Long: {sma_long.iloc[i]:.2f}, RSI: {rsi.iloc[i]:.2f}")
                
                # Calculate trend strength
                trend_strength = (sma_short.iloc[i] - sma_long.iloc[i]) / sma_long.iloc[i]
                
                # Buy signal conditions
                sma_cross_up = sma_short.iloc[i] > sma_long.iloc[i] and sma_short.iloc[i-1] <= sma_long.iloc[i-1]
                rsi_oversold = rsi.iloc[i] < self.rsi_lower
                strong_uptrend = trend_strength > self.min_trend_strength
                
                # Sell signal conditions
                sma_cross_down = sma_short.iloc[i] < sma_long.iloc[i] and sma_short.iloc[i-1] >= sma_long.iloc[i-1]
                rsi_overbought = rsi.iloc[i] > self.rsi_upper
                strong_downtrend = trend_strength < -self.min_trend_strength
                
                # Generate signals with position tracking
                if not self.in_position:
                    # Buy when price is rising and trend is strong
                    if (sma_cross_up and strong_uptrend) or (rsi_oversold and not strong_downtrend):
                        signals.append({
                            'ticker': ticker,
                            'timestamp': df.index[i],
                            'action': 'BUY',
                            'price': df['Close'].iloc[i],
                            'sma_short': sma_short.iloc[i],
                            'sma_long': sma_long.iloc[i],
                            'rsi': rsi.iloc[i]
                        })
                        self.in_position = True
                else:  # In position
                    # Sell when price is falling or trend is weak
                    if sma_cross_down or rsi_overbought or strong_downtrend:
                        signals.append({
                            'ticker': ticker,
                            'timestamp': df.index[i],
                            'action': 'SELL',
                            'price': df['Close'].iloc[i],
                            'sma_short': sma_short.iloc[i],
                            'sma_long': sma_long.iloc[i],
                            'rsi': rsi.iloc[i]
                        })
                        self.in_position = False
        
        return pd.DataFrame(signals)