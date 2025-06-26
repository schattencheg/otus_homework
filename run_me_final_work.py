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
            
            # Split data for validation
            split_idx = int(len(features) * 0.8)
            
            # Train momentum predictor with cross-validation
            print("\nTraining Momentum Predictor...")
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(self.momentum_predictor.model, features, momentum_labels, 
                                       cv=5, scoring='neg_mean_squared_error')
            print(f"Cross-validation MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Final fit on full data
            self.momentum_predictor.fit(features, momentum_labels)
            
            # Train regime classifier with silhouette analysis
            print("\nTraining Market Regime Classifier...")
            best_score = -1
            best_n_clusters = 3
            
            for n_clusters in range(2, 6):
                classifier = MarketRegimeClassifier(n_clusters=n_clusters)
                classifier.fit(features)
                labels = classifier.predict(features)
                score = silhouette_score(features, labels)
                print(f"Silhouette score with {n_clusters} clusters: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            
            self.regime_classifier = MarketRegimeClassifier(n_clusters=best_n_clusters)
            self.regime_classifier.fit(features)
            print(f"Selected {best_n_clusters} clusters with score {best_score:.4f}")
            
            # Initialize and train CNN
            self.cnn_predictor = CNNTradePredictor(
                input_channels=features.shape[1],
                sequence_length=self.sequence_length
            )
            
            # Prepare CNN training data
            y_cnn = np.where(momentum_labels > 0, 1, 0)  # Binary labels
            y_cnn = y_cnn[:len(sequence_tensor)]  # Align with sequence data
            y_cnn = torch.LongTensor(y_cnn)
            
            # Split data for CNN
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
            
            # Train CNN with early stopping
            print("\nTraining CNN...")
            self.cnn_predictor.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.cnn_predictor.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            best_state = None
            
            n_epochs = 100
            for epoch in range(n_epochs):
                # Training phase
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
                
                # Validation phase
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
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{n_epochs}]")
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.cnn_predictor.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Load best model
            if best_state is not None:
                self.cnn_predictor.load_state_dict(best_state)
            
            # Save models
            print("\nSaving models...")
            self.save_models()
            
            # Final validation
            self.cnn_predictor.eval()
            with torch.no_grad():
                final_val_correct = 0
                final_val_total = 0
                for batch_X, batch_y in val_loader:
                    outputs = self.cnn_predictor(batch_X)
                    _, predicted = outputs.max(1)
                    final_val_total += batch_y.size(0)
                    final_val_correct += predicted.eq(batch_y).sum().item()
                
                final_val_acc = 100. * final_val_correct / final_val_total
            
            return {
                'feature_importance': self.momentum_predictor.feature_importance,
                'n_regimes': self.regime_classifier.n_clusters,
                'momentum_cv_score': -cv_scores.mean(),
                'regime_silhouette_score': best_score,
                'cnn_val_accuracy': final_val_acc
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
        
    def calculate_trend(self, prices: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.Series:
        """Calculate trend direction using EMA crossover"""
        short_ema = prices.ewm(span=short_window, adjust=False).mean()
        long_ema = prices.ewm(span=long_window, adjust=False).mean()
        return (short_ema > long_ema).astype(int) * 2 - 1  # Returns 1 for uptrend, -1 for downtrend
    
    def generate_signals(self, data: pd.DataFrame, features_generator: FeaturesGenerator) \
            -> Tuple[np.ndarray, np.ndarray, float]:
        features, _, sequence_tensor = self.prepare_data(data, features_generator)
        
        # Get predictions from each model
        momentum_pred = self.momentum_predictor.predict(features)
        regime = self.regime_classifier.predict(features)
        regime_confidence = self.calculate_regime_confidence(features, regime)
        
        with torch.no_grad():
            trade_pred = torch.softmax(self.cnn_predictor(sequence_tensor), dim=1)
        
        # Calculate volatility and trend
        volatility = self.calculate_volatility(data['Returns'])
        volatility = volatility.fillna(method='bfill')
        trend = self.calculate_trend(data['Close'])
        
        # Define risk parameters
        max_position_size = 1.0
        min_confidence = 0.3  # Reduced confidence requirement
        max_volatility = 1.0  # Increased volatility cap
        momentum_threshold = 0.005  # 0.5% minimum expected return
        
        # Combine signals
        signals = np.zeros(len(features))
        position_sizes = np.zeros(len(features))
        
        for i in range(len(features)):
            # Skip if volatility is too high or regime confidence is too low
            if volatility.iloc[i] > max_volatility or regime_confidence[i] < min_confidence:
                continue
            
            # Skip low liquidity regimes
            if regime[i] == 0:  # Assuming 0 is low liquidity regime
                continue
            
            # Calculate base position size inversely proportional to volatility
            vol_factor = max(0, 1 - (volatility.iloc[i] / max_volatility))
            conf_factor = regime_confidence[i]
            base_size = max_position_size * vol_factor * conf_factor
            
            # Adjust position size based on momentum strength
            momentum_strength = abs(momentum_pred[i])
            if momentum_strength < momentum_threshold:
                continue
                
            position_sizes[i] = base_size * (momentum_strength / momentum_threshold)
            position_sizes[i] = min(position_sizes[i], max_position_size)  # Cap position size
            
            # Generate trade signals with stricter criteria
            if i >= self.sequence_length:
                pred_idx = i - self.sequence_length
                long_prob = trade_pred[pred_idx, 1].item()
                short_prob = trade_pred[pred_idx, 0].item()
                
                # Take trades with moderate conviction and trend consideration
                if (long_prob > 0.55 and 
                    momentum_pred[i] > momentum_threshold and 
                    (trend[i] > 0 or momentum_pred[i] > momentum_threshold * 2)):
                    signals[i] = 1
                elif (short_prob > 0.55 and 
                      momentum_pred[i] < -momentum_threshold and 
                      (trend[i] < 0 or momentum_pred[i] < -momentum_threshold * 2)):
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
    
    def backtest(self, data: pd.DataFrame, features_generator: FeaturesGenerator,
                 initial_balance: float = 100000) -> Dict[str, Any]:
        signals, position_sizes, avg_momentum = self.generate_signals(
            data, features_generator)
        
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        # Risk management parameters
        stop_loss_pct = 0.02  # Tighter stop loss
        take_profit_pct = 0.04  # Maintain good risk/reward
        trailing_stop_pct = 0.015  # Tighter trailing stop
        max_drawdown_pct = 0.10  # Stricter drawdown control
        
        # Track highest equity for trailing stop and drawdown
        highest_equity = initial_balance
        highest_price_since_entry = 0
        lowest_price_since_entry = float('inf')
        
        for i in range(len(signals)):
            price = data['Close'].iloc[i]
            signal = signals[i]
            size = position_sizes[i]
            
            current_value = balance
            if position != 0:
                unrealized_pnl = position * (price - entry_price)
                current_value += unrealized_pnl
                
                # Update price extremes since entry
                if position > 0:
                    highest_price_since_entry = max(highest_price_since_entry, price)
                    trailing_stop_price = highest_price_since_entry * (1 - trailing_stop_pct)
                    should_stop = price < trailing_stop_price
                else:  # Short position
                    lowest_price_since_entry = min(lowest_price_since_entry, price)
                    trailing_stop_price = lowest_price_since_entry * (1 + trailing_stop_pct)
                    should_stop = price > trailing_stop_price
                
                # Check stop loss, take profit, and trailing stop
                stop_loss_hit = abs((price - entry_price) / entry_price) > stop_loss_pct and \
                               np.sign(price - entry_price) != np.sign(position)
                take_profit_hit = abs((price - entry_price) / entry_price) > take_profit_pct and \
                                 np.sign(price - entry_price) == np.sign(position)
                
                # Close position if any exit condition is met
                if stop_loss_hit or take_profit_hit or should_stop:
                    balance = current_value
                    trades.append({
                        'type': 'close',
                        'price': price,
                        'pnl': unrealized_pnl,
                        'balance': balance,
                        'direction': 'long' if position > 0 else 'short',
                        'reason': 'stop_loss' if stop_loss_hit else \
                                 'take_profit' if take_profit_hit else 'trailing_stop'
                    })
                    position = 0
                    highest_price_since_entry = 0
                    lowest_price_since_entry = float('inf')
            
            # Check for regular signal-based position changes
            if position != 0 and signal != np.sign(position):
                pnl = position * (price - entry_price)
                balance += pnl
                trades.append({
                    'type': 'close',
                    'price': price,
                    'pnl': pnl,
                    'balance': balance,
                    'direction': 'long' if position > 0 else 'short',
                    'reason': 'signal'
                })
                position = 0
                highest_price_since_entry = 0
                lowest_price_since_entry = float('inf')
            
            # Open new position if we don't have one and drawdown is acceptable
            current_drawdown = 1 - current_value / highest_equity
            if position == 0 and signal != 0 and current_drawdown < max_drawdown_pct:
                position = signal * size
                entry_price = price
                highest_price_since_entry = price
                lowest_price_since_entry = price
                trades.append({
                    'type': 'open',
                    'direction': 'long' if signal > 0 else 'short',
                    'size': size,
                    'price': price,
                    'balance': balance
                })
            
            # Update equity curve and highest equity
            equity_curve.append(current_value)
            highest_equity = max(highest_equity, current_value)
        
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
        
        # Calculate strategy metrics
        total_return = (balance - initial_balance) / initial_balance
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        
        strategy_results = {
            'strategy': 'ML Ensemble',
            'final_balance': balance,
            'equity_curve': equity_curve,
            'trades': trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': len(trades),
            'avg_momentum': avg_momentum,
            'total_return': total_return,
            'annual_return': annual_return
        }
        
        # Calculate buy & hold results
        bh_results = self.calculate_buy_hold_returns(data, initial_balance)
        
        return {
            'strategy_results': strategy_results,
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
    """Plot backtest results comparing strategy vs buy & hold"""
    strategy_results = results['strategy_results']
    bh_results = results['buy_hold_results']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot equity curves
    ax1.plot(strategy_results['equity_curve'], label='ML Ensemble Strategy')
    ax1.plot([strategy_results['equity_curve'][0]] + 
             [bh_results['final_balance']] * (len(strategy_results['equity_curve'])-1),
             label='Buy & Hold', linestyle='--')
    ax1.set_title('Strategy Comparison')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot trade points
    trades = strategy_results['trades']
    for trade in trades:
        if trade['type'] == 'open':
            color = 'g' if trade['direction'] == 'long' else 'r'
            marker = '^' if trade['direction'] == 'long' else 'v'
            ax1.scatter(trades.index(trade), trade['balance'], 
                       color=color, marker=marker, s=100)
    
    # Plot performance metrics comparison
    metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']
    labels = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    strategy_values = [strategy_results[m] for m in metrics]
    bh_values = [bh_results[m] for m in metrics]
    
    ax2.bar(x - width/2, strategy_values, width, label='ML Ensemble Strategy')
    ax2.bar(x + width/2, bh_values, width, label='Buy & Hold')
    
    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('data', exist_ok=True)
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
    print("\nResults plot saved to data/strategy_results.png")

if __name__ == '__main__':
    main()