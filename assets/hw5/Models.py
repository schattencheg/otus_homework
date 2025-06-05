#%pip install backtesting
#%pip install nbformat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backtesting import Backtest, Strategy


class StockCNN(nn.Module):
    def __init__(self, input_channels, window_size, kernel_size_conv1=3, kernel_size_conv2=3):
        super(StockCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=kernel_size_conv1)  # Input: dynamic channels, Output: 16 channels
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size_conv2)  # Input: 16 channels, Output: 32 channels
        self.fc1 = nn.Linear(32 * (window_size - (kernel_size_conv1 - 1) -(kernel_size_conv2 - 1)), 64)  # Flattened size after conv
        self.fc2 = nn.Linear(64, 2)  # Output: 2 classes (Buy, Hold)

    def forward(self, x):
        # Input x shape: (batch_size, window_size, num_features)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, num_features, window_size)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output from convolution layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Return logits (raw output) for each class


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x


class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, h_n = self.gru(x)
        x = self.fc(h_n[-1])
        return x


class StockCNN_LSTM(nn.Module):
    def __init__(self, input_channels):
        super(StockCNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3)  # Input: dynamic channels, Output: 16 channels
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)  # Input: 16 channels, Output: 32 channels

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(64, 2)  # Output: 2 classes (Buy, Hold)

    def forward(self, x):
        # Input x shape: (batch_size, window_size, num_features)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, num_features, window_size) for Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Reshape for LSTM: [batch, seq_length, features]

        x, _ = self.lstm(x)  # Pass through LSTM
        x = x[:, -1, :]  # Take the last time step

        return self.fc1(x)  # Return logits for each class

