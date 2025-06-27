import os
import time
from typing import Any, Dict
import ccxt
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataPeriod, DataResolution
from assets.Position import Position
from assets.Order import Order
import credentials

pd.options.mode.chained_assignment = None  # default='warn'

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Configure matplotlib for non-interactive backend

matplotlib.use('Agg')

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
        
        # Print prediction statistics
        print(f"\nPrediction stats:")
        print(f"Mean: {predictions.mean():.3f}")
        print(f"Min: {predictions.min():.3f}")
        print(f"Max: {predictions.max():.3f}")
        
        # Add momentum to signals with more sensitive thresholds
        signals = []
        for pred in predictions:
            signal = 1 if pred > self.threshold else 0
            # More sensitive thresholds: 0.52 for buy, 0.48 for sell
            if signal != self.last_signal and (pred > 0.52 or pred < 0.48):
                self.last_signal = signal
            signals.append(self.last_signal)
        
        # Print signal statistics
        unique, counts = np.unique(signals, return_counts=True)
        print(f"\nSignal distribution:")
        for val, count in zip(unique, counts):
            print(f"Signal {val}: {count} occurrences ({count/len(signals)*100:.1f}%)")
        
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
    # Save plots instead of showing them when running in Docker
    plt.savefig(os.path.join('data', 'strategy_performance.png'))
    plt.close()

def flat_position(exchange: ccxt.Exchange):
    current_position: Position = Position(get_position(exchange))
    current_position_size = current_position.contracts * current_position.side
    if current_position_size != 0:
        close_side = 'sell' if current_position_size > 0 else 'buy'
        exchange.create_market_order(
            symbol='BTC/USDT',
            side=close_side,
            amount=abs(current_position_size),
            params={'type': 'future', 'reduceOnly': True}
        )
        logging.info(f"Closed existing {-current_position_size} position")
    # Check if position is closed
    current_position = Position(get_position(exchange))
    if current_position.contracts == 0:
        logging.info("Current position is FLAT")

def get_position(exchange: ccxt.Exchange) -> float:
    """Get current position size. Positive for long, negative for short, 0 for no position."""
    try:
        result = None
        positions = exchange.fetch_positions(['BTC/USDT'])
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                size = float(position['contracts'] or 0)
                side = 1 if position['side'] == 'long' else -1
                result = position
        return result
    except Exception as e:
        logging.error(f"Error fetching position: {e}")
        if 'not authenticated' in str(e).lower():
            raise ValueError("Authentication failed. Please check your API keys.")
        return 0.0

def execute_trade(exchange: ccxt.Exchange, side: str, amount: float):
    """Execute a trade on Binance Futures"""
    try:
        # Close any existing positions first
        current_position = Position(get_position(exchange))
        current_position_size = current_position.contracts * current_position.side
        if current_position_size != 0:
            close_side = 'sell' if current_position_size > 0 else 'buy'
            exchange.create_market_order(
                symbol='BTC/USDT',
                side=close_side,
                amount=abs(current_position_size),
                params={'type': 'future', 'reduceOnly': True}
            )
            logging.info(f"Closed existing {-current_position_size} position")
        
        # Open new position
        order_dict = exchange.create_market_order(
            symbol='BTC/USDT',
            side=side,
            amount=amount,
            params={'type': 'future'}
        )
        order = Order(order_dict)
        logging.info(f"Opened new {side} position of size {amount}")
    except Exception as e:
        logging.error(f"Error executing {side} trade: {e}")
        if 'insufficient balance' in str(e).lower():
            logging.error("Insufficient balance to execute trade")
        elif 'not authenticated' in str(e).lower():
            raise ValueError("Authentication failed. Please check your API keys.")
        raise

def save_model(model: nn.Module, metadata: Dict[str, Any], filepath: str):
    """Save model state and metadata to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    torch.save(state, filepath)

def run_train(data_provider: DataProvider, features_generator: FeaturesGenerator):
    """Run backtest mode: train model and evaluate performance"""
    print("Running backtest mode...")
    
    # Load and prepare data
    print("Loading data...")
    data_provider.data_load()
    
    if not data_provider.data:
        print("No cached data found, requesting new data...")
        data_provider.data_request()
    
    if not data_provider.data or 'BTC/USDT' not in data_provider.data:
        raise RuntimeError("Failed to load data. Please check your internet connection and API access.")
    
    data = data_provider.data['BTC/USDT']
    data.columns = [col.capitalize() for col in data.columns]
    
    # Train and evaluate model
    model, results = train_and_evaluate(data, features_generator)
    
    # Save model and results
    save_results(model, results, data)
    
    return model, results

def run_live_trading(model_path: str, data_provider: DataProvider, features_generator: FeaturesGenerator):
    """Run live trading mode using saved model"""
    print("Running live trading mode...")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('data', 'trading.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Run backtest mode first.")
    
    logging.info(f"Loading model from {model_path}")
    state = torch.load(model_path, weights_only=False)
    metadata = state['metadata']
    
    model = CNNModel(
        input_channels=metadata['input_channels'],
        sequence_length=metadata['sequence_length']
    )
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    logging.info(f"Model loaded successfully")

    # Initialize exchange connection
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
    
    # Initialize exchange with testnet URLs
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True,
            'test': True  # Use test network
        },
        'urls': {
            'api': {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
            },
            'test': {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
            }
        }
    })
    
    # Test connection and account access
    try:
        logging.info('Switching to sandbox mode')
        exchange.set_sandbox_mode(True)
        balance = exchange.fetch_balance()
        logging.info(f"Successfully connected to Binance Futures testnet")
        logging.info(f"Initial balance: {balance['USDT']['free']} USDT")
    except Exception as e:
        logging.error(f"Failed to connect to Binance Futures testnet: {e}")
        logging.error("Please ensure you're using testnet API keys from https://testnet.binancefuture.com/")
        raise
    
    # Start trading loop
    strategy = TradingStrategy(
        model=model,
        sequence_length=metadata['sequence_length'],
        threshold=metadata['threshold']
    )

    flat_position(exchange)
    
    while True:
        try:
            # Get latest data
            data_provider.data_refresh()
            if data_provider.has_new_data:
                data = data_provider.data['BTC/USDT']
                new_data = data.iloc[-1][['Open','High','Low','Close']]
                data.columns = [col.capitalize() for col in data.columns]
                
                # Generate features
                features_df, _ = features_generator.prepare_features(data)
                features = torch.FloatTensor(features_df.values[-metadata['sequence_length']:]).unsqueeze(0).transpose(1, 2)
                
                # Generate trading signal
                signal = strategy.generate_signals(features)[0]
                
                # Execute trade
                current_position: Position = Position(get_position(exchange))
                current_position_size = current_position.contracts * current_position.side
                event_taken = False
                if signal == 1 and current_position_size <= 0:
                    execute_trade(exchange, 'buy', 0.01)
                    logging.info("Opened LONG position")
                    event_taken = True
                elif signal == 0 and current_position_size >= 0:
                    execute_trade(exchange, 'sell', 0.01)
                    logging.info("Opened SHORT position")
                    event_taken = True
                # Log status
                if event_taken:
                    balance = exchange.fetch_balance()
                    total_balance = balance['total']['USDT']
                    logging.info(f"Balance: {total_balance:.2f} USDT, Signal: {signal}, Position: {current_position_size:.5f}")
            time.sleep(60)  # Wait 1 minute
            
        except KeyboardInterrupt:
            logging.info("Stopping live trading...")
            break
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            time.sleep(60)

def train_and_evaluate(data: pd.DataFrame, features_generator: FeaturesGenerator) -> tuple:
    """Train model and evaluate performance"""
    # Generate features
    features_df, feature_names = features_generator.prepare_features(data)
    
    # Create labels based on future returns with stronger signals
    future_returns = data['Close'].pct_change(periods=3).shift(-3).iloc[:-3].values
    
    # Create binary labels: 1 for significant up moves, 0 for significant down moves
    # Ignore very small moves by setting threshold
    threshold = 0.005  # 0.5% threshold for significant moves
    labels = np.zeros_like(future_returns)
    labels[future_returns > threshold] = 1  # Strong up moves
    labels[future_returns < -threshold] = 0  # Strong down moves
    labels[np.abs(future_returns) <= threshold] = 0  # Small moves treated as sell signals
    
    # Align labels with features
    labels = labels[len(labels)-len(features_df):]
    
    # Prepare data for CNN
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.values, labels, test_size=0.2, shuffle=False
    )
    
    # Convert to PyTorch tensors and reshape for CNN
    sequence_length = 20
    n_features = X_train.shape[1]
    
    # Prepare sequences
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
    
    # Train model
    model = CNNModel(input_channels=n_features, sequence_length=sequence_length)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train.reshape(-1, 1)),
        batch_size=64,
        shuffle=True
    )
    
    print("Training CNN model...")
    train_model(model, train_loader, criterion, optimizer)
    
    # Run backtest
    strategy = TradingStrategy(model, sequence_length=sequence_length)
    results = strategy.backtest(data.iloc[-len(X_test)-sequence_length:], X_test)
    
    return model, results

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer):
    """Train the CNN model"""
    n_epochs = 100
    best_loss = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
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

def save_results(model: nn.Module, results: dict, data: pd.DataFrame):
    """Save model, metadata, and performance plots"""
    metadata = {
        'input_channels': model.input_channels,
        'sequence_length': model.sequence_length,
        'threshold': 0.45,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'performance': {
            'final_balance': results['final_balance'],
            'return': (results['final_balance'] / 100000 - 1) * 100,
            'n_trades': len(results['trades'])
        }
    }
    
    model_path = os.path.join('models', 'cnn_trader.pth')
    save_model(model, metadata, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save performance plot
    plot_strategy_performance(data, results)

def run_backtest(model_path: str, data_provider: DataProvider, features_generator: FeaturesGenerator):
    """Run backtest mode using saved model on historical data"""
    print("Loading saved model...")
    try:
        state = torch.load(model_path, weights_only=False)
        print("Model state keys:", state.keys())
        metadata = state['metadata']
        print("Metadata:", metadata)
        
        model = CNNModel(
            input_channels=metadata['input_channels'],
            sequence_length=metadata['sequence_length']
        )
        print(f"Model architecture:\n{model}")
        
        # Check if model state dict exists
        if 'model_state_dict' not in state:
            print("Error: model_state_dict not found in saved state")
            if 'model_state' in state:
                print("Found 'model_state' instead, using that")
                model.load_state_dict(state['model_state'])
            else:
                raise KeyError("No model state found in saved file")
        else:
            model.load_state_dict(state['model_state_dict'])
        
        # Verify model parameters are loaded
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel has {param_count} trainable parameters")
        
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using 'python run_me_hw7.py train'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please ensure the model was trained and saved correctly")
        sys.exit(1)
    
    print("Loading historical data...")
    data_provider.data_load()
    data = data_provider.data[data_provider.tickers[0]]
    
    # Generate features
    features_df, _ = features_generator.prepare_features(data)
    features_array = features_df.values
    
    # Create sequences for CNN input
    sequence_length = metadata['sequence_length']
    sequences = []
    for i in range(len(features_array) - sequence_length + 1):
        sequence = features_array[i:i + sequence_length]
        sequences.append(sequence)
    
    # Convert to tensor with shape (batch_size, n_features, sequence_length)
    X = torch.FloatTensor(sequences)
    X = X.transpose(1, 2)  # Shape: (n_samples, n_channels, sequence_length)
    
    # Run backtest
    strategy = TradingStrategy(
        model,
        sequence_length=metadata['sequence_length'],
        threshold=metadata['threshold']
    )
    
    print("Running backtest...")
    results = strategy.backtest(data, X)
    
    # Plot and save results
    plot_strategy_performance(data, results)
    print(f"\nBacktest Results:")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Return: {(results['final_balance'] / 100000 - 1) * 100:.2f}%")
    print(f"Number of Trades: {len(results['trades'])}")

def main():
    # Initialize components
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.MINUTE_01,
        period=DataPeriod.YEAR_01,
        skip_dashboard=True
    )
    features_generator = FeaturesGenerator()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='CNN Trading Strategy')
    parser.add_argument('mode', choices=['train', 'live', 'backtest'],
                        help='Run mode: train, backtest or live (trade)')
    args = parser.parse_args()
    
    model_path = os.path.join('models', 'cnn_trader.pth')
    
    if args.mode == 'train':
        run_train(data_provider, features_generator)
    elif args.mode == 'live':
        run_live_trading(model_path, data_provider, features_generator)
    elif args.mode == 'backtest':
        run_backtest(model_path, data_provider, features_generator)
    


if __name__ == '__main__':
    # Remove hardcoded API keys - they should be set in environment variables
    if 'BINANCE_API_KEY' not in os.environ or 'BINANCE_API_SECRET' not in os.environ:
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        print("Get your testnet API keys from https://testnet.binancefuture.com/")
        sys.exit(1)
        
    main()
