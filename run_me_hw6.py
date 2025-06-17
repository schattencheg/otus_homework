from datetime import datetime
from enum import Enum
import os
import platform
from tqdm import tqdm

# Clear terminal at start
os.system('cls' if platform.system() == 'Windows' else 'clear')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataResolution
from assets.hw6.SimpleBacktester import SimpleBacktester


# Neural Network Models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(SimpleLSTM, self).__init__()
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
        super(SimpleCNN, self).__init__()
        print(f"CNN init - input_channels: {input_channels}, window_size: {window_size}")
        
        # First convolution layer
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        # Second convolution layer
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate size after convolutions and pooling
        # First conv reduces size by 2 (kernel_size=3)
        conv1_size = window_size - 2
        # Second conv reduces size by 2
        conv2_size = conv1_size - 2
        # MaxPool reduces size by half
        pooled_size = conv2_size // 2
        # Final flattened size
        self.final_size = pooled_size * 64
       
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_size, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        # Pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten and reshape
        x = x.reshape(x.size(0), -1)
        
        # Update final_size if needed
        if self.final_size != x.size(1):
            print(f"Warning: Expected size {self.final_size} but got {x.size(1)}")
            self.final_size = x.size(1)
            # Recreate the linear layer with correct size
            self.fc1 = nn.Linear(self.final_size, 64)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleVotingModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleVotingModel, self).__init__()
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


def prepare_data(ticker='BTC/USDT', resolution=DataResolution.DAY_01):
    # Load data
    data_provider = DataProvider(tickers=[ticker], skip_dashboard=True, resolution=resolution)
    data_provider.data_load()
    data_provider.data_refresh()
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

def train_models(X_train, y_train, window_size, num_features, device):
    # Prepare sequence data for neural networks
    X_seq, y_seq = create_sequences(X_train.values, y_train.values, window_size)
    X_tensor = torch.FloatTensor(X_seq).to(device)
    y_tensor = torch.LongTensor(y_seq).to(device)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Prepare data for voting model
    X_vote_tensor = torch.FloatTensor(X_train.values).to(device)
    y_vote_tensor = torch.LongTensor(y_train.values).to(device)
    vote_dataset = TensorDataset(X_vote_tensor, y_vote_tensor)
    vote_loader = DataLoader(vote_dataset, batch_size=32, shuffle=True)

    # Initialize models and move to device
    lstm_model = SimpleLSTM(num_features).to(device)
    cnn_model = SimpleCNN(num_features, window_size).to(device)
    voting_model = SimpleVotingModel(num_features).to(device)

    # Train LSTM
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters())
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    
    print("Training LSTM...")
    for epoch in tqdm(range(20), desc="LSTM epochs"):  # Increased epochs
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

    # Train CNN with correct input shape
    optimizer = torch.optim.Adam(cnn_model.parameters())
    best_loss = float('inf')
    no_improve = 0
    
    print("\nTraining CNN...")
    for epoch in tqdm(range(20), desc="CNN epochs"):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Reshape input for CNN: (batch, seq_len, features) -> (batch, features, seq_len)
            batch_X = batch_X.transpose(1, 2)
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
    
    print("\nTraining Voting Model...")
    for epoch in tqdm(range(20), desc="Voting Model epochs"):  # Increased epochs
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
    
    # Prepare sequence data for validation
    X_val_seq, y_val_seq = create_sequences(X_val_norm.values, y_val.values, window_size)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train models
    print("\nTraining models...")
    lstm_model, cnn_model, voting_model = train_models(X_train_norm, y_train, window_size, num_features, device)

    # Evaluate models on validation set
    print("\nEvaluating models on validation set...")
    with torch.no_grad():
        # Move validation data to device
        X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
        X_val_tensor_cnn = X_val_tensor.transpose(1, 2)
        y_val_tensor = torch.LongTensor(y_val_seq).to(device)
        X_vote_val_tensor = torch.FloatTensor(X_val_norm.values).to(device)
        y_vote_val_tensor = torch.LongTensor(y_val.values).to(device)
        
        # LSTM evaluation
        lstm_model.eval()
        lstm_pred = lstm_model(X_val_tensor)
        lstm_pred = torch.argmax(lstm_pred, dim=1)
        lstm_acc = (lstm_pred == y_val_tensor).float().mean().item()
        print(f"LSTM Accuracy: {lstm_acc:.2%}")

        # CNN evaluation
        cnn_model.eval()
        cnn_pred = cnn_model(X_val_tensor_cnn)
        cnn_pred = torch.argmax(cnn_pred, dim=1)
        cnn_acc = (cnn_pred == y_val_tensor).float().mean().item()
        print(f"CNN Accuracy: {cnn_acc:.2%}")

        # Voting model evaluation
        voting_model.eval()
        voting_pred = voting_model(X_vote_val_tensor)
        voting_pred = torch.argmax(voting_pred, dim=1)
        voting_acc = (voting_pred == y_vote_val_tensor).float().mean().item()
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
