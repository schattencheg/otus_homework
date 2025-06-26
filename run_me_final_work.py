import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import joblib

from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataPeriod, DataResolution

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

class MomentumPredictor:
    """Predicts short-term price momentum using GradientBoosting"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=4
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=[f'feature_{i}' for i in range(X.shape[1])]
        ).sort_values(ascending=False)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str) -> None:
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }, path)
    
    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importance = data['feature_importance']

class CNNTradePredictor(nn.Module):
    """CNN model for predicting trade entry points"""
    
    def __init__(self, input_channels: int, sequence_length: int):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flatten_size = 64 * (sequence_length // 2)
        
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 outputs for short/long
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        x = x.view(-1, self.flatten_size)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class MarketRegimeClassifier:
    """Identifies market regimes using K-Means clustering"""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = MinMaxScaler()
        
    def fit(self, X: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def save(self, path: str) -> None:
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler
        }, path)
    
    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.kmeans = data['kmeans']
        self.scaler = data['scaler']

class TradingStrategy:
    """Main trading strategy combining multiple models"""
    
    def __init__(self):
        self.momentum_predictor = MomentumPredictor()
        self.cnn_predictor = None  # Will be initialized during training
        self.regime_classifier = MarketRegimeClassifier()
        
        self.sequence_length = 20
        self.position_size = 1.0  # Base position size
        
    def prepare_data(self, data: pd.DataFrame, features_generator: FeaturesGenerator) \
            -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        # Generate features
        features, feature_names = features_generator.prepare_features(data)
        
        # Align data
        common_index = features.index.intersection(data.index)
        features = features.loc[common_index]
        data = data.loc[common_index]
        
        # Prepare momentum labels (5-day returns)
        momentum_labels = data['Returns'].rolling(5).sum().shift(-5)
        
        # Remove NaN values
        valid_idx = ~np.isnan(momentum_labels)
        features = features[valid_idx]
        momentum_labels = momentum_labels[valid_idx]
        
        # Prepare CNN data
        sequence_data = []
        for i in range(len(features) - self.sequence_length):
            sequence = features.iloc[i:i+self.sequence_length].values
            sequence_data.append(sequence)
        
        if not sequence_data:
            raise ValueError("Not enough data points for sequence generation")
            
        sequence_tensor = torch.FloatTensor(np.array(sequence_data))
        sequence_tensor = sequence_tensor.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        return features.values, momentum_labels.values, sequence_tensor
    
    def train(self, data: pd.DataFrame, features_generator: FeaturesGenerator) -> Dict[str, Any]:
        try:
            features, momentum_labels, sequence_tensor = self.prepare_data(data, features_generator)
            
            print(f"Training data shapes:")
            print(f"Features: {features.shape}")
            print(f"Momentum labels: {momentum_labels.shape}")
            print(f"Sequence tensor: {sequence_tensor.shape}")
            
            # Train momentum predictor
            self.momentum_predictor.fit(features, momentum_labels)
            
            # Train regime classifier
            self.regime_classifier.fit(features)
            
            # Initialize and train CNN
            self.cnn_predictor = CNNTradePredictor(
                input_channels=features.shape[1],
                sequence_length=self.sequence_length
            )
            
            # Prepare CNN training data
            y_cnn = np.where(momentum_labels > 0, 1, 0)  # Binary labels
            y_cnn = y_cnn[:len(sequence_tensor)]  # Align with sequence data
            y_cnn = torch.LongTensor(y_cnn)
            
            # Train CNN
            self.cnn_predictor.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.cnn_predictor.parameters())
            
            n_epochs = 50
            batch_size = 32
            dataset = TensorDataset(sequence_tensor, y_cnn)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            print("\nTraining CNN...")
            for epoch in range(n_epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.cnn_predictor(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}")
            
            # Save models
            print("\nSaving models...")
            self.save_models()
            
            return {
                'feature_importance': self.momentum_predictor.feature_importance,
                'n_regimes': self.regime_classifier.n_clusters
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def generate_signals(self, data: pd.DataFrame, features_generator: FeaturesGenerator) \
            -> Tuple[np.ndarray, np.ndarray, float]:
        features, _, sequence_tensor = self.prepare_data(data, features_generator)
        
        # Get predictions from each model
        momentum_pred = self.momentum_predictor.predict(features)
        regime = self.regime_classifier.predict(features)
        
        with torch.no_grad():
            trade_pred = torch.softmax(self.cnn_predictor(sequence_tensor), dim=1)
        
        # Combine signals
        signals = np.zeros(len(features))
        position_sizes = np.ones(len(features)) * self.position_size
        
        for i in range(len(features)):
            # Skip low liquidity regimes
            if regime[i] == 0:  # Assuming 0 is low liquidity regime
                continue
                
            # Use momentum for position sizing
            position_sizes[i] *= abs(momentum_pred[i])
            
            # Use CNN for trade direction
            if i >= self.sequence_length:
                pred_idx = i - self.sequence_length
                if trade_pred[pred_idx, 1] > 0.6:  # Long
                    signals[i] = 1
                elif trade_pred[pred_idx, 0] > 0.6:  # Short
                    signals[i] = -1
        
        return signals, position_sizes, np.mean(momentum_pred)
    
    def backtest(self, data: pd.DataFrame, features_generator: FeaturesGenerator,
                 initial_balance: float = 100000) -> Dict[str, Any]:
        signals, position_sizes, avg_momentum = self.generate_signals(
            data, features_generator)
        
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        for i in range(len(signals)):
            price = data['Close'].iloc[i]
            signal = signals[i]
            size = position_sizes[i]
            
            # Close position
            if position != 0 and signal != np.sign(position):
                pnl = position * (price - entry_price)
                balance += pnl
                trades.append({
                    'type': 'close',
                    'price': price,
                    'pnl': pnl,
                    'balance': balance,
                    'direction': 'long' if position > 0 else 'short'
                })
                position = 0
            
            # Open position
            if position == 0 and signal != 0:
                position = signal * size
                entry_price = price
                trades.append({
                    'type': 'open',
                    'direction': 'long' if signal > 0 else 'short',
                    'size': size,
                    'price': price,
                    'balance': balance
                })
            
            # Update equity curve
            current_value = balance
            if position != 0:
                current_value += position * (price - entry_price)
            equity_curve.append(current_value)
        
        # Close any remaining position at the end
        if position != 0:
            price = data['Close'].iloc[-1]
            pnl = position * (price - entry_price)
            balance += pnl
            trades.append({
                'type': 'close',
                'price': price,
                'pnl': pnl,
                'balance': balance,
                'direction': 'long' if position > 0 else 'short'
            })
            equity_curve.append(balance)
        
        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        closed_trades = [t for t in trades if t['type'] == 'close']
        winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0
        
        return {
            'final_balance': equity_curve[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': len([t for t in trades if t['type'] == 'close']),
            'equity_curve': equity_curve,
            'trades': trades,
            'avg_momentum': avg_momentum
        }
    
    def save_models(self) -> None:
        self.momentum_predictor.save('models/momentum_predictor.joblib')
        self.regime_classifier.save('models/regime_classifier.joblib')
        torch.save(self.cnn_predictor.state_dict(), 'models/cnn_predictor.pth')
    
    def load_models(self) -> None:
        self.momentum_predictor.load('models/momentum_predictor.joblib')
        self.regime_classifier.load('models/regime_classifier.joblib')
        
        # Load CNN model
        features_shape = joblib.load('models/momentum_predictor.joblib')['scaler'].n_features_in_
        self.cnn_predictor = CNNTradePredictor(features_shape, self.sequence_length)
        self.cnn_predictor.load_state_dict(torch.load('models/cnn_predictor.pth'))

def plot_results(results: Dict[str, Any]) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot equity curve
    ax1.plot(results['equity_curve'])
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True)
    
    # Plot trade points
    for trade in results['trades']:
        if trade['type'] == 'open':
            color = 'g' if trade['direction'] == 'long' else 'r'
            ax1.axvline(x=results['trades'].index(trade), color=color, alpha=0.2)
    
    # Print metrics
    metrics_text = f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
    metrics_text += f"Max Drawdown: {results['max_drawdown']:.2%}\n"
    metrics_text += f"Win Rate: {results['win_rate']:.2%}\n"
    metrics_text += f"Number of Trades: {results['n_trades']}"
    
    ax2.text(0.05, 0.5, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/strategy_results.png')
    plt.close()

def main():
    # Initialize components
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01
    )
    features_generator = FeaturesGenerator()
    
    # Load and process data
    data_provider.data_load()
    df = data_provider.data_processed['BTC/USDT']
    
    # Create and train strategy
    strategy = TradingStrategy()
    training_results = strategy.train(df, features_generator)
    
    print("\nTop feature importance:")
    print(training_results['feature_importance'].head())
    
    # Run backtest
    results = strategy.backtest(df, features_generator)
    
    print(f"\nBacktest Results:")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['n_trades']}")
    
    # Plot results
    plot_results(results)
    print("\nResults plot saved to data/strategy_results.png")

if __name__ == '__main__':
    main()