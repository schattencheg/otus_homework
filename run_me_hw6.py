import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataResolution


# Neural Network Models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # 2 classes: buy (1) or not buy (0)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * ((window_size - 2) // 2), 2)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape to (batch, channels, sequence)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class SimpleVotingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class SimpleBacktester:
    def __init__(self, data, features, lstm_model, cnn_model, voting_model, window_size):
        self.data = data
        self.features = features
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.voting_model = voting_model
        self.window_size = window_size
        self.positions = []
        self.trades = []
        self.cash = 1000000
        self.commission = 0.002
        
    def run(self):
        for i in range(self.window_size, len(self.data)):
            current_data = self.data.iloc[:i+1]
            signal = self._get_signal(current_data)
            
            if signal == 1 and not self.positions:  # Buy
                price = current_data.iloc[-1]['Close']
                size = (self.cash * (1 - self.commission)) / price
                self.positions.append({
                    'entry_time': current_data.index[-1],
                    'entry_price': price,
                    'size': size
                })
                self.cash -= price * size * (1 + self.commission)
                
            elif signal == 0 and self.positions:  # Sell
                position = self.positions.pop()
                price = current_data.iloc[-1]['Close']
                self.cash += position['size'] * price * (1 - self.commission)
                self.trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_data.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'pnl': (price/position['entry_price'] - 1) * 100  # percentage
                })
        
        return pd.DataFrame(self.trades)
    
    def _get_signal(self, data):
        features = self.features.loc[data.index[-self.window_size:]].values
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        last_features = torch.FloatTensor(features[-1]).unsqueeze(0)

        with torch.no_grad():
            lstm_pred = torch.softmax(self.lstm_model(features_tensor), dim=1)
            cnn_pred = torch.softmax(self.cnn_model(features_tensor), dim=1)
            voting_pred = torch.softmax(self.voting_model(last_features), dim=1)
        
        ensemble_pred = (lstm_pred.numpy() + cnn_pred.numpy() + voting_pred.numpy()) / 3
        return 1 if ensemble_pred[0][1] > 0.5 else 0


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
    
    # Create target variable (simple example: 1 if next day's return is positive)
    returns = data['Close'].pct_change().shift(-1)
    y = (returns > 0).astype(int)
    
    return feature_matrix[:-1], y[:-1], data  # Remove last row as it has NaN in target

def create_sequences(features, targets, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

def train_models(X_train, y_train, window_size, num_features):
    # Prepare sequence data for neural networks
    X_seq, y_seq = create_sequences(X_train, y_train, window_size)
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
    
    for epoch in range(5):  # 5 epochs for example
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Train CNN
    optimizer = torch.optim.Adam(cnn_model.parameters())
    for epoch in range(5):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = cnn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Train Voting Model
    optimizer = torch.optim.Adam(voting_model.parameters())
    for epoch in range(5):
        for batch_X, batch_y in vote_loader:
            optimizer.zero_grad()
            outputs = voting_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return lstm_model, cnn_model, voting_model

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
    
    # Split data (70% train, 30% test)
    train_size = int(0.7 * len(features))
    X_train = features[:train_size]
    y_train = targets[:train_size]
    test_data = price_data.iloc[train_size:]
    # Create a DataFrame with features for test data
    test_features = pd.DataFrame(features[train_size:], index=test_data.index)
    
    # Train models
    lstm_model, cnn_model, voting_model = train_models(X_train, y_train, window_size, num_features)
    
    # Run backtest on test data
    backtester = SimpleBacktester(test_data, test_features, lstm_model, cnn_model, voting_model, window_size)
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
