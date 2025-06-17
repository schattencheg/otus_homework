from assets.DataProvider import DataProvider
from assets.enums import DataResolution, DataPeriod
from pats_cuda import train_bull_detector
import datetime as dt
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize data provider for BTC/USD with 1-minute data for the last month
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.MINUTE_01,
        period=DataPeriod.MONTH_01
    )
    
    # Request data
    print("Loading data...")
    data_provider.data_request()
    
    if not data_provider.data:
        print("Failed to load data!")
        return
        
    print(f"Loaded {len(data_provider.data['BTC/USDT'])} data points")
    
    # Train model on CPU with 5 epochs
    logging.info("\nStarting CNN training on CPU...")
    model = train_bull_detector(data_provider, use_cuda=False, epochs=5)
    logging.info("\nTraining complete!")
    
    # Save the trained model
    torch.save(model.state_dict(), 'bull_pattern_model.pth')
    logging.info("Model saved to bull_pattern_model.pth")

if __name__ == '__main__':
    main()
