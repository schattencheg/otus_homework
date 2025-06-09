import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from enum import Enum
from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataResolution


# Neural Network Models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size//2, 2)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.bn1(x)
        x = self.dropout1(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.bn2(x)
        x = self.dropout2(x)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate final size after convolutions and pooling
        conv_size = window_size - 4  # After two conv layers with kernel_size=3
        pooled_size = conv_size // 2  # After MaxPool1d(2)
        self.final_size = pooled_size * 64
        
        self.fc1 = nn.Linear(self.final_size, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        # Input shape: (batch, channels, sequence)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleVotingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


class SimpleBacktester:
    def __init__(self, data, features, lstm_model, cnn_model, voting_model):
        self.data = data
        self.features = features
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.voting_model = voting_model
        
        # Initialize portfolio
        self.cash = 100000  # Starting with $100k
        self.positions = []  # List to track open positions
        self.trades = []  # List to track completed trades
        
        # Risk parameters
        self.max_positions = 5  # Max number of concurrent positions
        self.max_position_size = 2.0  # Maximum position size in BTC
        self.base_stop_loss_pct = 0.02  # Base stop loss at 2%
        self.trailing_stop_pct = 0.01  # 1% trailing stop
        self.risk_per_trade = 0.03  # Risk 3% per trade
        self.commission = 0.001  # 0.1% commission per trade
        
        # Performance tracking
        self.peak_value = 100000
        self.max_drawdown = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Technical parameters
        self.window_size = 60  # Lookback window for features
        
        # Initialize performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        
        # Calculate ATR for dynamic stop losses
        self.atr = self.calculate_atr(14)
        
        # Initialize performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_value = self.cash
    
    def calculate_atr(self, period):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    def get_dynamic_stops(self, current_price, current_time):
        # Get current ATR value
        current_atr = self.atr.loc[current_time]
        atr_multiplier = 2.0
        
        # Calculate dynamic stop loss (2-3 ATR)
        stop_loss_pct = min(self.base_stop_loss_pct * 1.5,
                           (current_atr * atr_multiplier) / current_price)
        
        # Calculate dynamic take profit (1.5-2x stop loss)
        take_profit_pct = stop_loss_pct * 2
        
        return stop_loss_pct, take_profit_pct
    
    def calculate_volatility(self, window=20):
        returns = self.data['Close'].pct_change()
        return returns.rolling(window=window).std()
        
    def calculate_position_size(self, signal_strength, current_price, current_vol):
        # Base position size as percentage of portfolio
        risk_per_trade = 0.03  # 3% risk per trade
        account_value = self.cash + sum(pos['size'] * current_price for pos in self.positions)
        base_size = risk_per_trade * signal_strength
        
        # Adjust for volatility (less impact)
        vol_percentile = current_vol / self.data['Close'].pct_change().std()
        vol_factor = 1 / (1 + vol_percentile * 0.5)  # Less reduction for volatility
        
        # Calculate target position value
        target_value = base_size * account_value * vol_factor
        size = target_value / current_price
        
        # Ensure minimum trade size of $1000
        min_size = 1000 / current_price
        size = max(size, min_size)
        
        # Cap at max position size
        size = min(size, self.max_position_size)
        
        # Round to 3 decimal places
        return round(size, 3)
        
    def run(self):
        volatility = self.calculate_volatility()
        self.trades = []  # Initialize trades list
        print("Starting backtest with cash:", self.cash)
        
        for i in range(self.window_size, len(self.data)):
            current_data = self.data.iloc[:i+1]
            current_price = current_data.iloc[-1]['Close']
            current_time = current_data.index[-1]
            current_vol = volatility.iloc[i] if i < len(volatility) else volatility.iloc[-1]
            
            # Update portfolio metrics
            portfolio_value = self.cash + sum(pos['size'] * current_price for pos in self.positions)
            print(f"\nTime: {current_time}, Price: {current_price:.2f}, Portfolio: {portfolio_value:.2f}")
            
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            else:
                drawdown = (self.peak_value - portfolio_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Get signal strength (0 to 1)
            signal_strength = self._get_signal(current_data)
            
            # Open new position if we have a signal
            if signal_strength is not None and signal_strength > 0:
                # Calculate position size based on risk and signal strength
                risk_amount = self.cash * self.risk_per_trade
                target_value = risk_amount * signal_strength * 10  # Scale up the position size
                
                # Adjust for volatility (reduce position size when volatility is high)
                volatility_factor = 1 / (1 + volatility)
                target_value = target_value * volatility_factor
                
                # Enforce minimum trade size
                target_value = max(target_value, 1000)  # Minimum $1000 position
                
                # Cap by maximum position size
                target_value = min(target_value, self.max_position_size)
                
                # Calculate number of units to buy
                position_size = round(target_value / current_price, 3)  # Round to 3 decimals
                
                # Check if we can afford it
                cost = position_size * current_price
                if cost <= self.cash:
                    # Calculate stop loss and take profit levels
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                    
                    # Open the position
                    self.positions.append({
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'high_water_mark': current_price
                    })
                    
                    # Deduct the cost plus commission
                    self.cash -= cost * (1 + self.commission)
                    print(f"Opened position: {position_size:.3f} units at {current_price:.2f}")
                    print(f"Cost: ${cost:.2f}, Remaining cash: ${self.cash:.2f}")
                    print(f"Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}")
                else:
                    print(f"Not enough cash (${self.cash:.2f}) for position costing ${cost:.2f}")
        
        # Close any remaining positions at the end
        if self.positions:
            final_price = self.data.iloc[-1]['Close']
            final_time = self.data.index[-1]
            
            for pos in self.positions:
                pnl = (final_price - pos['entry_price']) * pos['size']
                self.trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': final_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'size': pos['size'],
                    'pnl': pnl,
                    'return': pnl / (pos['entry_price'] * pos['size'])
                })
                self.cash += final_price * pos['size'] * (1 - self.commission)
            
            self.positions = []
        
        return pd.DataFrame(self.trades)
    
    def check_stops(self, data):
        current_price = data.iloc[-1]['Close']
        current_time = data.index[-1]
        closed_positions = []
        positions_to_remove = []
        
        # Check each position for stop loss, take profit, or trailing stop
        for pos in self.positions:
            # Update high water mark
            if current_price > pos['high_water_mark']:
                pos['high_water_mark'] = current_price
                # Update trailing stop
                pos['stop_loss'] = max(pos['stop_loss'],
                                     current_price * (1 - self.trailing_stop_pct))
            
            # Check if we hit stop loss or take profit
            if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                # Calculate PnL
                pnl = (current_price - pos['entry_price']) * pos['size']
                
                # Record the trade
                closed_positions.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'size': pos['size'],
                    'pnl': pnl,
                    'return': pnl / (pos['entry_price'] * pos['size'])
                })
                
                # Update cash
                self.cash += current_price * pos['size'] * (1 - self.commission)
                
                # Mark position for removal
                positions_to_remove.append(pos)
        
        # Remove closed positions
        for pos in positions_to_remove:
            self.positions.remove(pos)
        
        return closed_positions
    
    def _get_signal(self, data):
        # Set models to evaluation mode
        self.lstm_model.eval()
        self.cnn_model.eval()
        self.voting_model.eval()
        
        # Get recent price data for trend confirmation
        recent_close = data['Close'].iloc[-5:]  # Shorter lookback
        price_trend = recent_close.iloc[-1] > recent_close.mean()
        print(f"\nPrice trend: {price_trend}")
        
        # Calculate momentum indicators
        returns = data['Close'].pct_change()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Calculate MACD
    def check_stops(self, data):
        current_price = data.iloc[-1]['Close']
        current_time = data.index[-1]
        closed_positions = []
        positions_to_remove = []
        
        # Check each position for stop loss, take profit, or trailing stop
        for pos in self.positions:
            # Update high water mark
            if current_price > pos['high_water_mark']:
                pos['high_water_mark'] = current_price
                # Update trailing stop
                pos['stop_loss'] = max(pos['stop_loss'],
                                     current_price * (1 - self.trailing_stop_pct))
            
            # Check if we hit stop loss or take profit
            if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                # Calculate PnL
                pnl = (current_price - pos['entry_price']) * pos['size']
                
                # Record the trade
                closed_positions.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'size': pos['size'],
                    'pnl': pnl,
                    'return': pnl / (pos['entry_price'] * pos['size'])
                })
                
                # Update cash
                self.cash += current_price * pos['size'] * (1 - self.commission)
                
                # Mark position for removal
                positions_to_remove.append(pos)
        
        # Remove closed positions
        for pos in positions_to_remove:
            self.positions.remove(pos)
        
        return closed_positions
    
    def _get_signal(self, data):
        # Set models to evaluation mode
        self.lstm_model.eval()
        self.cnn_model.eval()
        self.voting_model.eval()
        
        # Get recent price data for trend confirmation
        recent_close = data['Close'].iloc[-5:]  # Shorter lookback
        price_trend = recent_close.iloc[-1] > recent_close.mean()
        print(f"\nPrice trend: {price_trend}")
        
        # Calculate momentum indicators
        returns = data['Close'].pct_change()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1.iloc[-1] - exp2.iloc[-1]
        signal = (exp1 - exp2).ewm(span=9, adjust=False).mean()
        macd_signal = signal.iloc[-1]
        
        # More lenient momentum conditions
        momentum_bullish = (
            rsi > 40 or  # RSI above 40 (even more lenient)
            macd > macd_signal or  # MACD crossover
            returns.iloc[-1] > 0  # Recent positive return
        )
        print(f"RSI: {rsi:.2f}, MACD: {macd:.2f}, MACD Signal: {macd_signal:.2f}")
        print(f"Momentum bullish: {momentum_bullish}")
        
        # Get recent features for prediction
        features = self.features.loc[data.index[-self.window_size:]].values
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        last_features = torch.FloatTensor(features[-1]).unsqueeze(0)
        
        with torch.no_grad():
            # LSTM prediction
            lstm_input = features_tensor.repeat(2, 1, 1)
            lstm_pred = torch.softmax(self.lstm_model(lstm_input), dim=1)[0]
            
            # CNN prediction (reshape input for CNN)
            cnn_input = features_tensor.transpose(1, 2)  # (batch, seq_len, features) -> (batch, features, seq_len)
            cnn_pred = torch.softmax(self.cnn_model(cnn_input), dim=1)[0]
            
            # Voting model prediction
            vote_input = last_features.repeat(2, 1)
            voting_pred = torch.softmax(self.voting_model(vote_input), dim=1)[0]
        
        # Weight models based on their validation accuracy
        lstm_weight = 0.25
        cnn_weight = 0.30
        vote_weight = 0.45
        
        # Calculate weighted ensemble prediction
        ensemble_pred = (
            lstm_pred.numpy() * lstm_weight +
            cnn_pred.numpy() * cnn_weight +
            voting_pred.numpy() * vote_weight
        )
        confidence = ensemble_pred[1]  # Probability of positive class
        print(f"Model confidence: {confidence:.2f}")
        
        # Even more lenient signal generation
        if price_trend or momentum_bullish:  # Only need one confirmation
            if confidence > 0.40:  # Lower threshold
                print("Taking full position")
                return 1.0
            elif confidence > 0.35:
                print("Taking 75% position")
                return 0.75
            elif confidence > 0.30:
                print("Taking 50% position")
                return 0.5
        
        print("No position taken")
        return 0


def prepare_data(ticker='BTC/USDT', resolution=DataResolution.DAY_01):
    # Load data
    data_provider = DataProvider(tickers=[ticker], skip_dashboard=True, resolution=resolution)
    data_provider.data_load()
    if ticker not in data_provider.data:
        data_provider.data_request()
        data_provider.clean_data()
    data = data_provider.data[ticker]

    # Generate features
    features_gen = FeaturesGenerator()
    feature_matrix, feature_names = features_gen.prepare_features(data)
    
    # Add market regime features
    sma_50 = data['Close'].rolling(window=50).mean()
    sma_200 = data['Close'].rolling(window=200).mean()
    feature_matrix['trend'] = (sma_50 > sma_200).astype(float)
    feature_matrix['trend_strength'] = ((sma_50 - sma_200) / sma_200 * 100)
    
    # Add volatility features
    returns = data['Close'].pct_change()
    feature_matrix['volatility_20'] = returns.rolling(window=20).std()
    feature_matrix['volatility_50'] = returns.rolling(window=50).std()
    
    # Add momentum features
    feature_matrix['momentum_14'] = returns.rolling(window=14).sum()
    feature_matrix['momentum_30'] = returns.rolling(window=30).sum()
    
    # Generate forward returns for multiple horizons
    forward_returns = {}
    horizons = [1, 3, 5, 7]  # Multiple time horizons
    
    for horizon in horizons:
        # Calculate forward returns
        fwd_ret = data['Close'].pct_change(horizon).shift(-horizon)
        # Calculate forward volatility
        fwd_vol = fwd_ret.rolling(window=horizon).std()
        # Calculate risk-adjusted returns
        risk_adj_ret = fwd_ret / fwd_vol
        forward_returns[horizon] = risk_adj_ret
    
    # Combine signals from multiple horizons
    combined_signal = pd.Series(0, index=feature_matrix.index)
    for horizon in horizons:
        # Weight longer horizons more
        weight = horizon / sum(horizons)
        threshold = 0.2 / weight  # Adjust threshold based on horizon
        # Align the forward returns with feature matrix index
        aligned_returns = forward_returns[horizon].reindex(feature_matrix.index)
        combined_signal += (aligned_returns > threshold).astype(int) * weight
    
    # Convert to binary classification with more stringent criteria
    y = (combined_signal > 0.5).astype(int)
    
    # Clean up any NaN values
    feature_matrix = feature_matrix.ffill().fillna(0)
    
    # Remove last few samples that don't have complete forward data
    valid_mask = ~y.isnull()
    
    # Ensure all features are numeric
    feature_matrix = feature_matrix.astype(float)
    
    return feature_matrix[valid_mask], y[valid_mask], data

def create_sequences(features, targets, window_size):
    if len(features) <= window_size:
        raise ValueError(f"Not enough samples ({len(features)}) for window size {window_size}")
    
    # Create sequences using numpy operations for better performance
    n_samples = len(features) - window_size
    n_features = features.shape[1]
    
    X = np.zeros((n_samples, window_size, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        X[i] = features[i:i + window_size]
        y[i] = targets[i + window_size]
    
    return X, y

def train_models(X_train, y_train, window_size, num_features):
    # Prepare sequence data for neural networks
    X_seq, y_seq = create_sequences(X_train.values, y_train.values, window_size)
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.LongTensor(y_seq)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Prepare data for voting model
    X_vote_tensor = torch.FloatTensor(X_train.values)
    y_vote_tensor = torch.LongTensor(y_train.values)
    vote_dataset = TensorDataset(X_vote_tensor, y_vote_tensor)
    vote_loader = DataLoader(vote_dataset, batch_size=32, shuffle=True)

    # Initialize models
    lstm_model = SimpleLSTM(num_features)
    cnn_model = SimpleCNN(num_features, window_size)
    voting_model = SimpleVotingModel(num_features)

    # Train LSTM
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters())
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    
    for epoch in range(20):  # Increased epochs
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Train CNN
    optimizer = torch.optim.Adam(cnn_model.parameters())
    best_loss = float('inf')
    no_improve = 0
    
    for epoch in range(20):  # Increased epochs
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = cnn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Train Voting Model
    optimizer = torch.optim.Adam(voting_model.parameters())
    best_loss = float('inf')
    no_improve = 0
    
    for epoch in range(20):  # Increased epochs
        total_loss = 0
        for batch_X, batch_y in vote_loader:
            optimizer.zero_grad()
            outputs = voting_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(vote_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return lstm_model, cnn_model, voting_model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / len(y)
    model.train()
    return accuracy

def normalize_features(train_features, val_features=None, test_features=None):
    # Calculate mean and std on training data
    train_mean = train_features.mean()
    train_std = train_features.std()
    train_std[train_std == 0] = 1  # Prevent division by zero
    
    # Normalize training data
    train_normalized = (train_features - train_mean) / train_std
    
    # If validation/test data is provided, normalize it using training statistics
    results = [train_normalized]
    if val_features is not None:
        val_normalized = (val_features - train_mean) / train_std
        results.append(val_normalized)
    if test_features is not None:
        test_normalized = (test_features - train_mean) / train_std
        results.append(test_normalized)
    
    return results

def main():
    # Parameters
    window_size = 30
    ticker = 'BTC/USDT'
    timeframe = DataResolution.DAY_01
    timeframe = DataResolution.MINUTE_01
    path = os.path.join('output','hw6')
    os.makedirs(path, exist_ok=True)

    # Prepare data
    features, targets, price_data = prepare_data(ticker, timeframe)
    num_features = features.shape[1]
    
    # Split data (60% train, 20% validation, 20% test)
    train_size = int(0.6 * len(features))
    val_size = int(0.2 * len(features))
    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_val = features[train_size:train_size+val_size]
    y_val = targets[train_size:train_size+val_size]
    test_data = price_data.iloc[train_size+val_size:]
    test_features = pd.DataFrame(features[train_size+val_size:], index=test_data.index)
    
    # Normalize features
    X_train_norm, X_val_norm, test_features_norm = normalize_features(X_train, X_val, test_features)
    
    # Convert validation data to tensors
    X_val_seq, y_val_seq = create_sequences(X_val_norm.values, y_val.values, window_size)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.LongTensor(y_val_seq)
    
    # Train models
    lstm_model, cnn_model, voting_model = train_models(X_train_norm, y_train, window_size, num_features)
    
    # Evaluate models on validation set
    print("\nValidation Accuracies:")
    lstm_acc = evaluate_model(lstm_model, X_val_tensor, y_val_tensor)
    cnn_acc = evaluate_model(cnn_model, X_val_tensor, y_val_tensor)
    voting_acc = evaluate_model(voting_model, torch.FloatTensor(X_val.values), torch.LongTensor(y_val.values))
    print(f"LSTM Accuracy: {lstm_acc:.2%}")
    print(f"CNN Accuracy: {cnn_acc:.2%}")
    print(f"Voting Model Accuracy: {voting_acc:.2%}")
    
    # Create and run backtester
    backtester = SimpleBacktester(test_data, test_features_norm, lstm_model, cnn_model, voting_model)
    trades = backtester.run()
    
    # Print results
    if len(trades) > 0:
        print(f"\nBacktesting Results:")
        print(f"Number of trades: {len(trades)}")
        print(f"Average PnL per trade: {trades['pnl'].mean():.2f}%")
        print(f"Total return: {trades['pnl'].sum():.2f}%")
        print(f"Win rate: {(trades['pnl'] > 0).mean() * 100:.1f}%")
        
        # Save trades to CSV
        trades.to_csv(os.path.join(path, 'ensemble_trades.csv'))
        print("\nTrades saved to 'ensemble_trades.csv'")
    else:
        print("No trades were executed during the test period.")


if __name__ == "__main__":
    main()
