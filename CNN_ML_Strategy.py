import numpy as np
import pandas as pd
import ccxt
import time
from typing import List, Tuple, Optional
from assets.DataProvider import DataProvider
from sklearn.preprocessing import MinMaxScaler

class SimplePricePredictorModel:
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.weights = np.random.randn(sequence_length) / sequence_length  # Simple weighted average
        
    def predict(self, X: np.ndarray) -> float:
        # Simple weighted average of the sequence
        return np.dot(X, self.weights)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, learning_rate: float = 0.01):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                pred = self.predict(X[i])
                error = y[i] - pred
                # Update weights using gradient descent
                self.weights += learning_rate * error * X[i]
                total_loss += error ** 2
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X):.4f}')

class CNNTradingStrategy:
    def __init__(self, ticker: str = 'BTC/USD', sequence_length: int = 20):
        self.ticker = ticker
        self.sequence_length = sequence_length
        # Create DataProvider with Kraken exchange
        self.data_provider = DataProvider(tickers=[ticker])
        exchange = ccxt.kraken({
            'enableRateLimit': True,  # Enable built-in rate limiter
            'rateLimit': 3000  # Time between requests in milliseconds
        })
        self.data_provider.data_loader.exchange = exchange
        self.model = SimplePricePredictorModel(sequence_length=sequence_length)
        self.scaler = MinMaxScaler()
        self._data = None  # Cache for the data
        
    def prepare_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Get data from provider if not cached
        if self._data is None:
            self._data = self.data_provider.get_data(self.ticker)
        
        if self._data is None or len(self._data) < self.sequence_length:
            print(f"Not enough data available for {self.ticker}")
            return None
            
        df = self._data
        
        # Create features using just closing prices for simplicity
        close_prices = df['Close'].values  # Note the capital C
        scaled_prices = self.scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - self.sequence_length):
            X.append(scaled_prices[i:i + self.sequence_length])
            # Target: actual next price (for regression)
            y.append(scaled_prices[i + self.sequence_length])
        
        if not X or not y:
            print("Failed to create training sequences")
            return None
            
        return np.array(X), np.array(y)
    
    def train(self, epochs: int = 50):
        data = self.prepare_data()
        if data is None:
            print("Cannot train - no data available")
            return False
            
        X, y = data
        self.model.train(X, y, epochs=epochs)
        return True
    
    def predict_next(self, window_data: np.ndarray) -> Optional[float]:
        if len(window_data) != self.sequence_length:
            print(f"Window data must be of length {self.sequence_length}")
            return None
            
        # Scale the window data
        scaled_window = self.scaler.transform(window_data.reshape(-1, 1)).flatten()
        prediction = self.model.predict(scaled_window)
        
        # Inverse transform to get actual price prediction
        return self.scaler.inverse_transform([[prediction]])[0][0]
    
    def generate_signals(self, threshold: float = 0.01) -> Optional[pd.Series]:
        # Use cached data
        if self._data is None:
            self._data = self.data_provider.get_data(self.ticker)
            
        if self._data is None or len(self._data) < self.sequence_length:
            print(f"Not enough data available for {self.ticker}")
            return None
            
        df = self._data
            
        signals = pd.Series(index=df.index, data=0)
        close_prices = df['Close'].values
        
        for i in range(self.sequence_length, len(close_prices)):
            window = close_prices[i-self.sequence_length:i]
            pred_price = self.predict_next(window)
            
            if pred_price is None:
                continue
                
            current_price = close_prices[i-1]
            price_change = (pred_price - current_price) / current_price
            
            if price_change > threshold:
                signals.iloc[i] = 1  # Buy signal
            elif price_change < -threshold:
                signals.iloc[i] = -1  # Sell signal
        
        return signals

def test_strategy():
    # Test with BTC/USD on Kraken
    strategy = CNNTradingStrategy('BTC/USD', sequence_length=20)
    
    print("Training the model...")
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            if strategy.train(epochs=50):
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All retry attempts failed")
                return
    print("\nGenerating trading signals...")
    signals = strategy.generate_signals(threshold=0.01)
    
    if signals is not None:
        total_signals = len(signals[signals != 0])
        buy_signals = len(signals[signals == 1])
        sell_signals = len(signals[signals == -1])
        
        print(f"\nTrading Signal Summary:")
        print(f"Total signals generated: {total_signals}")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        
        # Test prediction
        df = strategy.data_provider.get_data('BTC/USD')
        if df is not None and len(df) >= 20:
            last_window = df['Close'].values[-20:]
            next_price = strategy.predict_next(last_window)
            current_price = df['Close'].values[-1]
            
            print(f"\nPrediction Test:")
            print(f"Current price: {current_price:.2f}")
            print(f"Predicted next price: {next_price:.2f}")
            print(f"Predicted change: {((next_price - current_price) / current_price * 100):.2f}%")

if __name__ == '__main__':
    test_strategy()
