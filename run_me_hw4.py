import logging
import sys
from datetime import datetime, timedelta
from assets.DataProvider import DataProvider
from assets.HW3_ML_Strategy import HW3_ML_Strategy
from assets.enums import DataPeriod, DataResolution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_data(data_provider, min_days=60):
    """Validate that we have enough data for training"""
    for ticker in data_provider.tickers:
        if ticker not in data_provider.data_processed:
            raise ValueError(f"No processed data available for {ticker}")
        
        df = data_provider.data_processed[ticker]
        if df is None or df.empty:
            raise ValueError(f"Empty dataset for {ticker}")
        
        # Check date range
        date_range = (df.index.max() - df.index.min()).days
        if date_range < min_days:
            raise ValueError(
                f"Insufficient data for {ticker}. Need at least {min_days} days, "
                f"got {date_range} days"
            )
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {ticker}: {missing_columns}"
            )

def main():
    try:
        # Initialize data provider with BTC/USDT data
        data_provider = DataProvider(
            tickers=['BTC/USDT'],
            resolution=DataResolution.DAY_01,
            period=DataPeriod.YEAR_05, ts = None, te = None)
        
        # Load and process data
        logging.info("Loading historical data...")
        data_provider.data_load()
        #data_provider.data_request()
        #data_provider.data_save()
        
        if not data_provider.data:
            logging.info("No cached data found, requesting from exchange...")
            data_provider.data_request()
            logging.info("Saving data to cache...")
            data_provider.data_save()
        
        # Validate data
        validate_data(data_provider)
        
        # Initialize ML strategy
        ml_strategy = HW3_ML_Strategy(
            data_provider,
            test_size=0.2,
            random_state=42,
            n_estimators=200,  # More trees for better stability
            max_depth=5,      # Prevent overfitting
            min_samples_split=10
        )
        
        for ticker in data_provider.tickers:
            logging.info(f"\nProcessing {ticker}...")
            
            try:
                # Train model and get predictions
                logging.info(f"Training ML strategy for {ticker}...")
                predictions = ml_strategy.train(ticker)
                
                # Create performance dashboard
                logging.info(f"Creating dashboard for {ticker}...")
                ml_strategy.dashboard_create(ticker, predictions)
                                
                # Save metrics
                logging.info(f"Saving metrics for {ticker}...")
                ml_strategy.save_metrics(ticker)
                
                logging.info(f"Strategy evaluation completed for {ticker}")
                logging.info("Results are saved in the 'results/ml_strategy' directory")

                ml_strategy.dashboard_show()
                
            except Exception as e:
                logging.error(f"Error processing {ticker}: {str(e)}")
                continue
    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
