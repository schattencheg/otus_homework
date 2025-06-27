import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    """Identifies market regimes using K-Means clustering with smoothing"""
    
    def __init__(self, n_clusters=3, smoothing_window=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = MinMaxScaler()
        self.smoothing_window = smoothing_window
        self.last_regimes = []
        
    def fit(self, X: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        raw_predictions = self.kmeans.predict(X_scaled)
        
        # Apply smoothing to regime predictions
        smoothed_predictions = np.copy(raw_predictions)
        for i in range(len(raw_predictions)):
            # Get recent regime history
            start_idx = max(0, i - self.smoothing_window)
            recent_regimes = raw_predictions[start_idx:i+1]
            
            # Only change regime if it's persistent in recent window
            if len(recent_regimes) > 0:
                current_regime = recent_regimes[-1]
                regime_counts = np.bincount(recent_regimes)
                most_common_regime = np.argmax(regime_counts)
                
                # Change regime only if the new regime is dominant
                if regime_counts[most_common_regime] >= len(recent_regimes) * 0.7:
                    smoothed_predictions[i] = most_common_regime
                elif i > 0:
                    # Maintain previous regime if no clear dominance
                    smoothed_predictions[i] = smoothed_predictions[i-1]
        
        return smoothed_predictions
    
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
        
        # Convert list of arrays to a single numpy array first
        sequence_data = np.array(sequence_data)
        sequence_tensor = torch.FloatTensor(sequence_data)
        sequence_tensor = sequence_tensor.transpose(1, 2)  # (batch, channels, seq_len)
        
        return features.values, momentum_labels.values, sequence_tensor
    
    def train(self, data: pd.DataFrame, features_generator: FeaturesGenerator) -> Dict[str, Any]:
        try:
            features, momentum_labels, sequence_tensor = self.prepare_data(data, features_generator)
            
            print(f"Training data shapes:")
            print(f"Features: {features.shape}")
            print(f"Momentum labels: {momentum_labels.shape}")
            print(f"Sequence tensor: {sequence_tensor.shape}")
            
            # Split data for validation
            split_idx = int(len(features) * 0.8)
            
            # Train momentum predictor with cross-validation
            print("\nTraining Momentum Predictor...")
            cv_scores = cross_val_score(self.momentum_predictor.model, features, momentum_labels, 
                                       cv=5, scoring='neg_mean_squared_error')
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()
            print(f"Cross-validation MSE: {cv_mse:.4f} (+/- {cv_std:.4f})")
            
            self.momentum_predictor.fit(features, momentum_labels)
            
            # Train regime classifier
            print("\nTraining Market Regime Classifier...")
            best_n_clusters = 2
            best_score = -1
            
            for n_clusters in range(2, 6):
                self.regime_classifier = MarketRegimeClassifier(n_clusters=n_clusters)
                self.regime_classifier.fit(features)
                labels = self.regime_classifier.predict(features)
                score = silhouette_score(features, labels)
                print(f"Silhouette score with {n_clusters} clusters: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            
            print(f"\nSelected {best_n_clusters} clusters with score {best_score:.4f}")
            
            # Train CNN
            print("\nTraining CNN...")
            # Initialize model
            self.cnn_predictor = CNNTradePredictor(features.shape[1], self.sequence_length)
            
            # Split data for CNN
            # Prepare binary labels for the entire sequence first
            y_cnn = np.where(momentum_labels > 0, 1, 0)  # Binary labels
            y_cnn = y_cnn[self.sequence_length:]  # Align with sequence data
            y_cnn = torch.LongTensor(y_cnn)
            
            # Split data
            train_size = int(0.8 * len(sequence_tensor))
            train_sequence = sequence_tensor[:train_size]
            train_labels = y_cnn[:train_size]
            val_sequence = sequence_tensor[train_size:]
            val_labels = y_cnn[train_size:]
            
            # Create data loaders
            batch_size = 32
            train_dataset = TensorDataset(train_sequence, train_labels)
            val_dataset = TensorDataset(val_sequence, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.cnn_predictor.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=5)
            
            # Training loop
            epochs = 100
            best_val_loss = float('inf')
            best_state = None
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                self.cnn_predictor.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.cnn_predictor(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                # Validation
                self.cnn_predictor.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.cnn_predictor(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                train_loss = train_loss / len(train_loader)
                val_loss = val_loss / len(val_loader)
                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}]")
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%    ")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.cnn_predictor.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                scheduler.step(val_loss)
            
            # Load best model
            if best_state is not None:
                self.cnn_predictor.load_state_dict(best_state)
            
            # Save models
            print("\nSaving models...")
            self.save_models()
            
            # Return training results
            return {
                'momentum_cv_score': cv_mse,
                'feature_importance': self.momentum_predictor.feature_importance,
                'n_regimes': best_n_clusters,
                'regime_silhouette_score': best_score,
                'cnn_val_accuracy': val_acc
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_regime_confidence(self, features: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate confidence in regime classification using distance to cluster centers"""
        distances = self.regime_classifier.kmeans.transform(self.regime_classifier.scaler.transform(features))
        confidences = 1 - (distances[np.arange(len(regime_labels)), regime_labels] / np.max(distances, axis=1))
        return confidences
        
    def calculate_trend(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """Calculate trend using exponential moving averages with longer window"""
        short_ema = prices.ewm(span=window, adjust=False).mean()
        long_ema = prices.ewm(span=window*2, adjust=False).mean()
        
        # Calculate trend strength
        trend_strength = (short_ema - long_ema) / long_ema
        
        # Return trend direction weighted by strength
        return pd.Series(np.sign(trend_strength) * np.minimum(abs(trend_strength) * 100, 1), 
                        index=prices.index)
    
    def generate_signals(self, data: pd.DataFrame, features_generator: FeaturesGenerator) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate trading signals using the ensemble of models"""
        features, _ = features_generator.prepare_features(data)
        
        # Align data
        common_index = features.index.intersection(data.index)
        features = features.loc[common_index]
        data = data.loc[common_index]
        
        # Convert features to numpy array for predictions
        features_array = features.values
        
        # Prepare sequence data for CNN
        sequence_data = []
        for i in range(len(features) - self.sequence_length):
            sequence = features.iloc[i:i+self.sequence_length].values
            sequence_data.append(sequence)
        
        if not sequence_data:
            raise ValueError("Not enough data points for sequence generation")
        
        sequence_data = np.array(sequence_data)
        sequence_tensor = torch.FloatTensor(sequence_data)
        sequence_tensor = sequence_tensor.transpose(1, 2)  # (batch, channels, seq_len)
        
        # Get predictions from each model
        momentum_pred = self.momentum_predictor.predict(features_array)
        regime = self.regime_classifier.predict(features_array)
        regime_confidence = self.calculate_regime_confidence(features_array, regime)
        
        with torch.no_grad():
            trade_pred = torch.softmax(self.cnn_predictor(sequence_tensor), dim=1)
        
        # Calculate volatility and trend
        volatility = self.calculate_volatility(data['Returns'], window=20)
        volatility = volatility.bfill()
        trend = self.calculate_trend(data['Close'], window=50)
        
        # Calculate longer-term trend strength
        sma_50 = data['Close'].rolling(window=50).mean()
        sma_200 = data['Close'].rolling(window=200).mean()
        trend_strength = ((sma_50 - sma_200) / sma_200).fillna(0)
        
        # Signal generation parameters
        max_position_size = 1.0
        min_confidence = 0.25
        max_volatility = 1.2
        momentum_threshold = 0.003
        
        signals = np.zeros(len(features))
        position_sizes = np.zeros(len(features))
        
        # Signal smoothing window
        signal_window = 5
        long_signals = np.zeros(signal_window)
        short_signals = np.zeros(signal_window)
        
        for i in range(len(features)):
            # Skip if volatility is too high or regime confidence is too low
            if volatility.iloc[i] > max_volatility or regime_confidence[i] < min_confidence:
                continue
            
            # Skip if in ranging regime and trend is weak
            if regime[i] == 0 and abs(trend_strength.iloc[i]) < 0.01:
                continue
            
            # Calculate position size factors
            vol_factor = max(0.2, 1 - (volatility.iloc[i] / max_volatility))
            conf_factor = regime_confidence[i] * (1 + abs(trend_strength.iloc[i]))
            base_size = max_position_size * vol_factor * conf_factor
            
            # Scale position size by momentum strength
            momentum_value = momentum_pred[i]
            momentum_strength = abs(momentum_value)
            
            if momentum_strength >= momentum_threshold:
                position_sizes[i] = base_size * (momentum_strength / momentum_threshold)
                position_sizes[i] = min(position_sizes[i], max_position_size)
            
            # Generate signals only after we have enough sequence data
            if i >= self.sequence_length:
                pred_idx = i - self.sequence_length
                long_prob = trade_pred[pred_idx, 1].item()
                short_prob = trade_pred[pred_idx, 0].item()
                
                # Update signal windows
                long_signals = np.roll(long_signals, -1)
                short_signals = np.roll(short_signals, -1)
                
                # Check signal conditions
                long_signal = (long_prob > 0.53 and 
                             momentum_value > momentum_threshold and 
                             trend.iloc[i] > 0)
                short_signal = (short_prob > 0.53 and 
                              momentum_value < -momentum_threshold and 
                              trend.iloc[i] < 0)
                
                long_signals[-1] = 1 if long_signal else 0
                short_signals[-1] = 1 if short_signal else 0
                
                # Generate final signals based on persistence
                if np.sum(long_signals) >= signal_window * 0.6 and trend_strength.iloc[i] > 0:
                    signals[i] = 1
                elif np.sum(short_signals) >= signal_window * 0.6 and trend_strength.iloc[i] < 0:
                    signals[i] = -1
        
        return signals, position_sizes, np.mean(momentum_pred)
        
        # Calculate volatility and trend with longer windows
        volatility = self.calculate_volatility(data['Returns'], window=20)  # Increased window
        volatility = volatility.bfill()
        trend = self.calculate_trend(data['Close'], short_window=20, long_window=50)  # Longer trend window
        
        # Calculate additional trend indicators
        sma_50 = data['Close'].rolling(window=50).mean()
        sma_200 = data['Close'].rolling(window=200).mean()
        trend_strength = ((sma_50 - sma_200) / sma_200).fillna(0)
        
        # Define risk parameters with reduced sensitivity
        max_position_size = 1.0
        min_confidence = 0.25  # Lower confidence requirement
        max_volatility = 1.2  # Higher volatility tolerance
        momentum_threshold = 0.003  # Lower momentum threshold
        
        # Combine signals
        signals = np.zeros(len(features))
        position_sizes = np.zeros(len(features))
        
        # Track signal persistence
        signal_window = 5
        long_signals = np.zeros(signal_window)
        short_signals = np.zeros(signal_window)
        
        for i in range(len(features)):
            # Basic volatility and regime checks
            if volatility.iloc[i] > max_volatility or regime_confidence[i] < min_confidence:
                continue
            
            # More sophisticated regime analysis
            if regime[i] == 0 and trend_strength.iloc[i] < 0.01:  # Low liquidity and weak trend
                continue
            
            # Position sizing with trend consideration
            vol_factor = max(0.2, 1 - (volatility.iloc[i] / max_volatility))  # Minimum 20% size
            conf_factor = regime_confidence[i] * (1 + abs(trend_strength.iloc[i]))
            base_size = max_position_size * vol_factor * conf_factor
            
            # Momentum analysis with trend alignment
            momentum_value = momentum_pred.iloc[i]
            momentum_strength = abs(momentum_value)
            
            if momentum_strength >= momentum_threshold:
                position_sizes[i] = base_size * (momentum_strength / momentum_threshold)
                position_sizes[i] = min(position_sizes[i], max_position_size)
            
            # Generate trade signals with trend following
            if i >= self.sequence_length:
                pred_idx = i - self.sequence_length
                long_prob = trade_pred[pred_idx, 1].item()
                short_prob = trade_pred[pred_idx, 0].item()
                
                # Update signal history
                long_signals = np.roll(long_signals, -1)
                short_signals = np.roll(short_signals, -1)
                
                # Calculate signal strength with trend alignment
                long_signal = (long_prob > 0.53 and 
                             momentum_value > momentum_threshold and 
                             trend.iloc[i] > 0)
                short_signal = (short_prob > 0.53 and 
                              momentum_value < -momentum_threshold and 
                              trend.iloc[i] < 0)
                
                long_signals[-1] = 1 if long_signal else 0
                short_signals[-1] = 1 if short_signal else 0
                
                # Generate signals only on persistent conditions
                if np.sum(long_signals) >= signal_window * 0.6 and trend_strength.iloc[i] > 0:
                    signals[i] = 1
                elif np.sum(short_signals) >= signal_window * 0.6 and trend_strength.iloc[i] < 0:
                    signals[i] = -1
        
        return signals, position_sizes, np.mean(momentum_pred)
    
    def calculate_buy_hold_returns(self, data: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
        """Calculate buy and hold strategy returns"""
        price_ratio = data['Close'].iloc[-1] / data['Close'].iloc[0]
        final_balance = initial_balance * price_ratio
        returns = data['Returns']
        
        # Calculate metrics
        total_return = (final_balance - initial_balance) / initial_balance
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        drawdowns = (data['Close'] / data['Close'].cummax() - 1)
        max_drawdown = drawdowns.min()
        
        return {
            'strategy': 'Buy & Hold',
            'final_balance': final_balance,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return
        }
    
    def backtest(self, data: pd.DataFrame, features_generator: FeaturesGenerator) -> Dict[str, Any]:
        """Run backtest of the strategy"""
        initial_balance = 10000
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_values = []  # Will store equity values aligned with data
    
        # Get signals and position sizes
        signals, position_sizes, avg_momentum = self.generate_signals(data, features_generator)
        
        # Calculate returns and volatility for Sharpe ratio
        returns = []
        
        # Run backtest
        for i in range(len(signals)):
            close = data['Close'].iloc[i]
            signal = signals[i]
            
            # Update position P&L
            if position != 0:
                unrealized_pnl = position * (close - entry_price)
                current_value = balance + unrealized_pnl
            else:
                current_value = balance
            
            equity_values.append(current_value)
            
            # Calculate return for this period
            if len(equity_values) > 1:
                period_return = (equity_values[-1] - equity_values[-2]) / equity_values[-2]
                returns.append(period_return)
            
            if signal != 0 and position == 0:
                # Open position
                position = signal
                entry_price = close
                position_size = position_sizes[i]
                trades.append({
                    'type': 'open',
                    'direction': 'long' if signal > 0 else 'short',
                    'price': close,
                    'size': position_size,
                    'balance': balance,
                    'time': data.index[i]
                })
            elif position != 0 and (signal == 0 or signal == -position):
                # Close position
                pnl = position * (close - entry_price)
                balance += pnl
                trades.append({
                    'type': 'close',
                    'price': close,
                    'pnl': pnl,
                    'balance': balance,
                    'time': data.index[i]
                })
                position = 0
                entry_price = 0
        # Calculate strategy metrics
        total_return = (balance - initial_balance) / initial_balance
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        
        # Create equity curve series
        equity_curve = pd.Series(equity_values, index=data.index[:len(equity_values)])
        
        # Calculate Sharpe ratio
        returns_array = np.array(returns)
        if len(returns_array) > 0:
            volatility = np.std(returns_array) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Calculate win rate
        closed_trades = [t for t in trades if t['type'] == 'close']
        winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0
        
        # Calculate buy & hold results
        bh_results = self.calculate_buy_hold_returns(data, initial_balance)
        
        # Return results
        return {
            'strategy_results': {
                'final_balance': balance,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'n_trades': len(trades),
                'equity_curve': equity_curve,
                'trades': trades,
                'price_data': data[['Close']]
            },
            'buy_hold_results': bh_results
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
    """Plot backtest results comparing strategy vs buy & hold using Plotly"""
    strategy_results = results['strategy_results']
    bh_results = results['buy_hold_results']
    
    # Create figure with three subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price Chart with Trades', 'Strategy Performance Comparison', 'Performance Metrics'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Get price data and dates
    price_data = strategy_results['price_data']
    dates = price_data.index
    
    # Plot price chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price_data['Close'],
            name='Price',
            line=dict(color='lightgray')
        ),
        row=1, col=1
    )
    
    # Plot equity curves
    equity_data = pd.DataFrame(strategy_results['equity_curve'])
    equity_dates = equity_data.index
    values = equity_data.values.flatten()
    
    fig.add_trace(
        go.Scatter(
            x=equity_dates,
            y=values,
            name='ML Strategy',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Buy & Hold line
    fig.add_trace(
        go.Scatter(
            x=[equity_dates[0], equity_dates[-1]],
            y=[bh_results['final_balance']] * 2,
            name='Buy & Hold',
            line=dict(color='gray', dash='dash')
        ),
        row=2, col=1
    )
    
    # Plot trade points on both price and equity charts
    trades = strategy_results['trades']
    long_entries_price = []
    short_entries_price = []
    long_exits_price = []
    short_exits_price = []
    long_entries_equity = []
    short_entries_equity = []
    long_exits_equity = []
    short_exits_equity = []
    
    # Create lookup for price data
    price_lookup = price_data['Close'].to_dict()
    
    # Track last trade direction to match exits with entries
    last_trade_direction = None
    
    for trade in trades:
        trade_time = pd.Timestamp(trade['time'])
        if trade_time not in price_lookup:
            continue
            
        if trade['type'] == 'open':
            last_trade_direction = trade['direction']
            if trade['direction'] == 'long':
                long_entries_price.append((trade_time, price_lookup[trade_time]))
                long_entries_equity.append((trade_time, trade['balance']))
            else:
                short_entries_price.append((trade_time, price_lookup[trade_time]))
                short_entries_equity.append((trade_time, trade['balance']))
        elif trade['type'] == 'close' and last_trade_direction:
            if last_trade_direction == 'long':
                long_exits_price.append((trade_time, price_lookup[trade_time]))
                long_exits_equity.append((trade_time, trade['balance']))
            else:
                short_exits_price.append((trade_time, price_lookup[trade_time]))
                short_exits_equity.append((trade_time, trade['balance']))
            last_trade_direction = None
    
    # Plot trade points on price chart
    if long_entries_price:
        dates, prices = zip(*long_entries_price)
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='markers',
            name='Long Entries',
            marker=dict(symbol='triangle-up', size=10, color='green'),
        ), row=1, col=1)
    
    if long_exits_price:
        dates, prices = zip(*long_exits_price)
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='markers',
            name='Long Exits',
            marker=dict(symbol='x', size=10, color='lightgreen'),
        ), row=1, col=1)
    
    if short_entries_price:
        dates, prices = zip(*short_entries_price)
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='markers',
            name='Short Entries',
            marker=dict(symbol='triangle-down', size=10, color='red'),
        ), row=1, col=1)
    
    if short_exits_price:
        dates, prices = zip(*short_exits_price)
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='markers',
            name='Short Exits',
            marker=dict(symbol='x', size=10, color='pink'),
        ), row=1, col=1)
    
    # Plot trade points on equity curve
    if long_entries_equity:
        dates, values = zip(*long_entries_equity)
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='markers',
            name='Long Entries',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            showlegend=False
        ), row=2, col=1)
    
    if long_exits_equity:
        dates, values = zip(*long_exits_equity)
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='markers',
            name='Long Exits',
            marker=dict(symbol='x', size=10, color='lightgreen'),
            showlegend=False
        ), row=2, col=1)
    
    if short_entries_equity:
        dates, values = zip(*short_entries_equity)
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='markers',
            name='Short Entries',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            showlegend=False
        ), row=2, col=1)
    
    if short_exits_equity:
        dates, values = zip(*short_exits_equity)
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='markers',
            name='Short Exits',
            marker=dict(symbol='x', size=10, color='pink'),
            showlegend=False
        ), row=2, col=1)
    
    # Plot performance metrics comparison
    metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']
    labels = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']
    
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[strategy_results[m] for m in metrics],
            name='ML Strategy',
            marker_color='blue'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[bh_results[m] for m in metrics],
            name='Buy & Hold',
            marker_color='gray'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Trading Strategy Results",
        title_x=0.5,
        barmode='group'
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Metrics", row=3, col=1)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    
    # Save interactive plot
    os.makedirs('data', exist_ok=True)
    fig.write_html('data/strategy_results.html')

def main():
    # Initialize components
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.DAY_01,
        period=DataPeriod.YEAR_01
    )
    features_generator = FeaturesGenerator()
    
    # Load and process data
    print("Loading and processing data...")
    data_provider.data_load()
    df = data_provider.data_processed['BTC/USDT']
    
    # Create and train strategy
    print("\nInitializing and training strategy...")
    strategy = TradingStrategy()
    training_results = strategy.train(df, features_generator)
    
    print("\nModel Training Results:")
    print("-" * 50)
    print("Momentum Predictor:")
    print(f"Mean Squared Error (CV): {training_results['momentum_cv_score']:.4f}")
    print("\nTop 5 Important Features:")
    print(training_results['feature_importance'].head())
    
    print("\nMarket Regime Classifier:")
    print(f"Number of Regimes: {training_results['n_regimes']}")
    print(f"Silhouette Score: {training_results['regime_silhouette_score']:.4f}")
    
    print("\nCNN Trade Predictor:")
    print(f"Validation Accuracy: {training_results['cnn_val_accuracy']:.2f}%")
    
    # Run backtest
    print("\nRunning backtest...")
    results = strategy.backtest(df, features_generator)
    
    print(f"\nML Strategy Results:")
    print("-" * 50)
    print(f"Final Balance: ${results['strategy_results']['final_balance']:,.2f}")
    print(f"Total Return: {results['strategy_results']['total_return']:.2%}")
    print(f"Annual Return: {results['strategy_results']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['strategy_results']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['strategy_results']['max_drawdown']:.2%}")
    print(f"Win Rate: {results['strategy_results']['win_rate']:.2%}")
    print(f"Number of Trades: {results['strategy_results']['n_trades']}")
    
    print(f"\nBuy & Hold Results:")
    print("-" * 50)
    print(f"Final Balance: ${results['buy_hold_results']['final_balance']:,.2f}")
    print(f"Total Return: {results['buy_hold_results']['total_return']:.2%}")
    print(f"Annual Return: {results['buy_hold_results']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['buy_hold_results']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['buy_hold_results']['max_drawdown']:.2%}")
    
    # Plot results
    plot_results(results)
    print("\nInteractive results plot saved to data/strategy_results.html")

if __name__ == '__main__':
    main()