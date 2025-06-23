import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

class PriceDataset(Dataset):
    def __init__(self, data, window_size=60, stride=1):
        """Dataset for price patterns
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            window_size (int): Number of candles to look at
            stride (int): Stride for sliding window
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.prepare_data()
    
    def prepare_data(self):
        # Create sequences and labels
        sequences = []
        labels = []
        
        for i in range(0, len(self.data) - self.window_size - 20, self.stride):
            # Get window of data
            window = self.data.iloc[i:i+self.window_size]
            
            # Calculate future returns (20 periods ahead)
            future_return = (self.data.iloc[i+self.window_size+20]['Close'] / 
                           self.data.iloc[i+self.window_size]['Close'] - 1)
            
            # Label as bullish if return > 1%
            label = 1 if future_return > 0.01 else 0
            
            # Normalize data in window
            normalized = (window - window.mean()) / window.std()
            sequences.append(normalized.values)
            labels.append(label)
        
        self.sequences = torch.FloatTensor(np.array(sequences))
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class BullPatternCNN(nn.Module):
    def __init__(self, input_channels=5, device='cpu'):
        """CNN for detecting bullish patterns
        Args:
            input_channels (int): Number of input features (OHLCV)
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        ).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(device)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for Conv1d
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, epochs=50, device='cpu', temperature=2.0):
    """Train the CNN model
    Args:
        model (BullPatternCNN): Model instance
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of training epochs
        device (str): 'cuda' or 'cpu'
        temperature (float): Softmax temperature scaling factor
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    
    logging.info(f'Starting training on {device}')
    for epoch in tqdm(range(epochs), desc='Training epochs'):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc='Training batches', leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            scaled_outputs = outputs / temperature
            probs = torch.softmax(scaled_outputs, dim=1)
            loss = criterion(probs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation batches', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        logging.info(f'Epoch {epoch+1}/{epochs}')
        logging.info(f'Train Loss: {train_loss/len(train_loader):.3f} | '
                  f'Train Acc: {100.*correct/total:.2f}%')
        logging.info(f'Val Loss: {val_loss/len(val_loader):.3f} | '
                  f'Val Acc: {100.*val_correct/val_total:.2f}%')

def prepare_data(data_provider, window_size=60, batch_size=32, train_split=0.8):
    """Prepare data for training
    Args:
        data_provider: Instance of DataProvider
        window_size (int): Size of input window
        batch_size (int): Training batch size
        train_split (float): Train/val split ratio
    Returns:
        train_loader, val_loader (DataLoader): Training and validation data loaders
    """
    # Get data from data provider
    df = data_provider.data['BTC/USDT']
    
    # Create dataset
    dataset = PriceDataset(df, window_size=window_size)
    
    # Split into train/val
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def train_bull_detector(data_provider, use_cuda=False, epochs=50):
    """Train bull pattern detector
    Args:
        data_provider: Instance of DataProvider
        use_cuda (bool): Whether to use CUDA
        epochs (int): Number of training epochs
    Returns:
        model (BullPatternCNN): Trained model
    """
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Prepare data
    train_loader, val_loader = prepare_data(data_provider)
    
    # Create and train model
    model = BullPatternCNN(device=device)
    train_model(model, train_loader, val_loader, epochs=epochs, device=device)
    
    return model