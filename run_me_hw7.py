import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from assets.DataProvider import DataProvider, DataResolution, DataPeriod
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataPeriod, DataResolution

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

class CNNModel(nn.Module):
    def __init__(self, input_channels: int, sequence_length: int):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        
        # Very simple CNN architecture
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Using pooling to reduce dimensionality
        self.pool = nn.MaxPool1d(2)
        
        # Calculate the size of flattened features
        self.flatten_size = 16 * (sequence_length // 2)  # After 1 pooling layer
        
        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape: (batch, channels, sequence_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, self.flatten_size)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x

class TradingStrategy:
    def __init__(self, model: CNNModel, sequence_length: int = 20, threshold: float = 0.45):
        self.model = model
        self.sequence_length = sequence_length
        self.threshold = threshold  # Lower threshold for more trades
        self.trades = []
        self.last_signal = 0  # For signal momentum
        
    def generate_signals(self, features: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features).numpy()
        
        # Add momentum to signals
        signals = []
        for pred in predictions:
            signal = 1 if pred > self.threshold else 0
            # Only change position if we have a strong signal or opposite direction
            if signal != self.last_signal and (pred > 0.55 or pred < 0.45):
                self.last_signal = signal
            signals.append(self.last_signal)
        
        return np.array(signals)
    
    def backtest(self, data: pd.DataFrame, features: torch.Tensor,
                 initial_balance: float = 100000) -> Dict[str, Any]:
        signals = self.generate_signals(features)
        
        balance = initial_balance
        position = 0
        trades = []
        equity_curve = [initial_balance]
        
        for i, signal in enumerate(signals):
            price = data['Close'].iloc[i + self.model.sequence_length]
            
            if signal and position == 0:  # Buy signal
                position = balance / price
                trades.append({
                    'type': 'buy',
                    'price': price,
                    'timestamp': data.index[i + self.model.sequence_length]
                })
            elif not signal and position > 0:  # Sell signal
                balance = position * price
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': price,
                    'timestamp': data.index[i + self.model.sequence_length]
                })
            
            # Update equity curve
            current_equity = balance if position == 0 else position * price
            equity_curve.append(current_equity)
        
        self.trades = trades
        return {
            'final_balance': current_equity,
            'return': (current_equity - initial_balance) / initial_balance * 100,
            'trades': trades,
            'equity_curve': equity_curve
        }

def plot_strategy_performance(data: pd.DataFrame, strategy_results: Dict[str, Any]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot price and trades
    ax1.plot(data.index, data['Close'], label='Price', alpha=0.7)
    
    for trade in strategy_results['trades']:
        if trade['type'] == 'buy':
            ax1.scatter(trade['timestamp'], trade['price'], color='green', marker='^', s=100)
        else:
            ax1.scatter(trade['timestamp'], trade['price'], color='red', marker='v', s=100)
    
    ax1.set_title('Price Chart with Trading Signals')
    ax1.legend()
    ax1.grid(True)
    
    # Plot equity curve
    ax2.plot(data.index[:len(strategy_results['equity_curve'])],
             strategy_results['equity_curve'], label='Equity Curve')
    ax2.set_title('Equity Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_model(model: nn.Module, metadata: Dict[str, Any], filepath: str):
    """Save model state and metadata to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    torch.save(state, filepath)

def main():
    # Initialize components with specific configuration
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.HOUR_01,
        period=DataPeriod.YEAR_01,
        skip_dashboard=True  # Skip dashboard creation to avoid potential errors
    )
    features_generator = FeaturesGenerator()
    
    # Try to load existing data first
    print("Loading data...")
    data_provider.data_load()
    
    if not data_provider.data:
        print("No cached data found, requesting new data...")
        data_provider.data_request()
    
    if not data_provider.data or 'BTC/USDT' not in data_provider.data:
        raise RuntimeError("Failed to load data. Please check your internet connection and API access.")
    
    # Get processed data for BTC/USDT
    data = data_provider.data['BTC/USDT']
    
    # Ensure column names are correct
    data.columns = [col.capitalize() for col in data.columns]
    
    # Generate features
    features_df, feature_names = features_generator.prepare_features(data)
    
    # Create labels based on future returns
    future_returns = data['Close'].pct_change(periods=3).shift(-3).iloc[:-3].values  # 3-period future returns
    labels = (future_returns > 0).astype(float)
    # Add label smoothing for small moves
    labels = np.where(np.abs(future_returns) > 0.01, labels, 0.5)
    labels = labels[len(labels)-len(features_df):]
    
    # Prepare data for CNN
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.values, labels, test_size=0.2, shuffle=False
    )
    
    # Convert to PyTorch tensors and reshape for CNN
    sequence_length = 20
    n_features = X_train.shape[1]
    
    # Prepare sequences for CNN
    X_train_seq = []
    X_test_seq = []
    y_train_seq = []
    y_test_seq = []
    
    for i in range(sequence_length, len(X_train)):
        X_train_seq.append(X_train[i-sequence_length:i])
        y_train_seq.append(y_train[i])
    
    for i in range(sequence_length, len(X_test)):
        X_test_seq.append(X_test[i-sequence_length:i])
        y_test_seq.append(y_test[i])
    
    X_train = torch.FloatTensor(np.array(X_train_seq)).transpose(1, 2)
    X_test = torch.FloatTensor(np.array(X_test_seq)).transpose(1, 2)
    y_train = torch.FloatTensor(np.array(y_train_seq))
    y_test = torch.FloatTensor(np.array(y_test_seq))
    
    # Initialize and train model with improved parameters
    model = CNNModel(input_channels=n_features, sequence_length=sequence_length)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    n_epochs = 100  # Increased epochs
    batch_size = 64  # Increased batch size
    train_loader = DataLoader(
        TensorDataset(X_train, y_train.reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=True
    )
    
    print("Training CNN model...")
    best_loss = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct_predictions += (predictions == batch_y).sum().item()
            total_predictions += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Initialize and run strategy
    strategy = TradingStrategy(model, sequence_length=sequence_length)
    results = strategy.backtest(data.iloc[-len(X_test)-sequence_length:], X_test)
    
    print(f"\nStrategy Performance:")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Return: {((results['final_balance'] / 100000 - 1) * 100):.2f}%")
    print(f"Number of Trades: {len(results['trades'])}")
    
    # Save model and metadata
    metadata = {
        'input_channels': n_features,
        'sequence_length': sequence_length,
        'threshold': strategy.threshold,
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'performance': {
            'final_balance': results['final_balance'],
            'return': ((results['final_balance'] / 100000 - 1) * 100),
            'n_trades': len(results['trades'])
        }
    }
    
    model_path = os.path.join('models', 'cnn_trader.pth')
    save_model(model, metadata, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot results
    plot_strategy_performance(data.iloc[-len(X_test)-sequence_length:], results)

if __name__ == '__main__':
    main()