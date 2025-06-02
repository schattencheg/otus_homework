from assets.DataProvider import DataProvider
import ccxt
import time

def test_data_loading():
    # Initialize DataProvider with Kraken exchange
    provider = DataProvider(tickers=['BTC/USD'])
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000
    })
    provider.data_loader.exchange = exchange

    # Try to get data multiple times
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}:")
            df = provider.get_data('BTC/USD')
            if df is not None:
                print("Data loaded successfully!")
                print("Data shape:", df.shape)
                print("\nFirst few rows:")
                print(df.head())
                print("\nLast few rows:")
                print(df.tail())
                return
        except Exception as e:
            print(f"Error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All retry attempts failed")

if __name__ == '__main__':
    test_data_loading()
