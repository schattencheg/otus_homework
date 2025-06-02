import ccxt

# Initialize Kraken exchange
kraken = ccxt.kraken()

# Load markets
kraken.load_markets()

# Print all BTC pairs
btc_pairs = [symbol for symbol in kraken.symbols if 'BTC' in symbol or 'XBT' in symbol]
print("Available BTC/XBT pairs on Kraken:")
for pair in sorted(btc_pairs):
    print(pair)
