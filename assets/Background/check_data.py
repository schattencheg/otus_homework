from assets.DataProvider import DataProvider
import ccxt

# Initialize DataProvider with Kraken exchange
provider = DataProvider(tickers=['BTC/USD'])
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'rateLimit': 3000
})
provider.data_loader.exchange = exchange

# Get data
df = provider.get_data('BTC/USD')
if df is not None:
    print("\nData columns:", df.columns)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData shape:", df.shape)
else:
    print("No data available")
