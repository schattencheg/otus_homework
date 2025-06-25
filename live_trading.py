import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any
import time
import ccxt
from datetime import datetime, timedelta

from assets.DataProvider import DataProvider, DataResolution, DataPeriod
from assets.FeaturesGenerator import FeaturesGenerator
from run_me_hw7 import CNNModel

class LiveTrader:
    def __init__(self, model_path: str, api_key: str = None, api_secret: str = None):
        # Load model and metadata
        self.model, self.metadata = self.load_model(model_path)
        self.model.eval()
        
        # Initialize components
        self.data_provider = DataProvider(
            tickers=['BTC/USDT'],
            resolution=DataResolution.HOUR_01,
            period=DataPeriod.MONTH_01,
            skip_dashboard=True
        )
        self.features_generator = FeaturesGenerator()
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'future',  # use futures
            },
            'enableRateLimit': True,
        })
        
        if api_key and api_secret:
            self.exchange.set_sandbox_mode(True)  # Enable testnet
        
        self.position = 0
        self.sequence_length = self.metadata.get('sequence_length', 20)
        self.threshold = self.metadata.get('threshold', 0.45)
    
    @staticmethod
    def load_model(model_path: str) -> tuple:
        """Load model and metadata from file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
        
        state = torch.load(model_path)
        metadata = state['metadata']
        
        model = CNNModel(
            input_channels=metadata['input_channels'],
            sequence_length=metadata['sequence_length']
        )
        model.load_state_dict(state['model_state_dict'])
        return model, metadata
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get latest market data"""
        self.data_provider.data_load()
        if not self.data_provider.data:
            self.data_provider.data_request()
        
        if not self.data_provider.data or 'BTC/USDT' not in self.data_provider.data:
            raise RuntimeError("Failed to load data")
        
        return self.data_provider.data['BTC/USDT']
    
    def prepare_features(self, data: pd.DataFrame) -> torch.Tensor:
        """Prepare features for prediction"""
        # Ensure column names are correct
        data.columns = [col.capitalize() for col in data.columns]
        
        # Generate features
        features_df, _ = self.features_generator.prepare_features(data)
        
        # Convert to tensor and reshape for CNN
        features = torch.FloatTensor(features_df.values[-self.sequence_length:])
        features = features.unsqueeze(0).transpose(1, 2)
        return features
    
    def get_position(self) -> float:
        """Get current position size"""
        try:
            positions = self.exchange.fetch_positions(['BTC/USDT'])
            for position in positions:
                if position['symbol'] == 'BTC/USDT':
                    return float(position['contracts'] or 0)
        except Exception as e:
            print(f"Error fetching position: {e}")
        return 0
    
    def execute_trade(self, signal: int):
        """Execute trade based on signal"""
        try:
            current_position = self.get_position()
            
            if signal == 1 and current_position <= 0:
                # Open long position
                self.exchange.create_market_buy_order(
                    symbol='BTC/USDT',
                    amount=0.01,  # Trade with 0.01 BTC
                    params={'type': 'future'}
                )
                logging.info(f"Opened LONG position")
                
            elif signal == 0 and current_position >= 0:
                # Close position or go short
                self.exchange.create_market_sell_order(
                    symbol='BTC/USDT',
                    amount=0.01,  # Trade with 0.01 BTC
                    params={'type': 'future'}
                )
                logging.info(f"Opened SHORT position")
                
        except Exception as e:
            print(f"Error executing trade: {e}")
    
    def run(self, interval_seconds: int = 3600):
        """Run live trading"""
        print("Starting live trading...")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                # Get latest data and generate prediction
                data = self.get_latest_data()
                features = self.prepare_features(data)
                
                with torch.no_grad():
                    prediction = self.model(features).item()
                
                # Generate trading signal
                signal = 1 if prediction > self.threshold else 0
                
                # Execute trade if needed
                self.execute_trade(signal)
                
                # Get account balance
                balance = self.exchange.fetch_balance()
                total_balance = balance['total']['USDT']
                logging.info(f"Current balance: {total_balance:.2f} USDT")
                logging.info(f"Current prediction: {prediction:.4f}")
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nStopping live trading...")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    # Load your API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    model_path = os.getenv('MODEL_PATH', 'models/cnn_trader.pth')
    
    if not api_key or not api_secret:
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('data', 'trading.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    model_path = 'models/cnn_trader.pth'
    trader = LiveTrader(model_path, api_key, api_secret)
    trader.run()

if __name__ == '__main__':
    main()
