from datetime import datetime, timedelta
from assets.HW5_ML_Strategy import HW5_ML_Strategy
import pandas as pd
import traceback

def analyze_trades(trades):
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    profitable_trades = 0
    total_profit = 0
    current_position = None
    
    for i in range(0, len(df)-1, 2):
        if i+1 < len(df):  # Make sure we have a pair of trades
            buy_price = df.iloc[i]['price'] if df.iloc[i]['type'] == 'buy' else df.iloc[i+1]['price']
            sell_price = df.iloc[i+1]['price'] if df.iloc[i+1]['type'] == 'sell' else df.iloc[i]['price']
            profit = sell_price - buy_price
            if profit > 0:
                profitable_trades += 1
            total_profit += profit
    
    total_trades = len(df) // 2
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'average_profit': avg_profit
    }

def main():
    # Test parameters
    symbols = ['BTC/USDT', 'ETH/USDT']  # Test multiple trading pairs
    timeframes = ['1h', '4h']          # Test multiple timeframes
    initial_balance = 10000            # Initial balance for backtesting
    
    # Date range for testing (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    results = []
    
    print(f"Starting strategy tests from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial balance: ${initial_balance:,.2f}\n")
    
    # Test strategy with different combinations
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                print(f"\nTesting {symbol} on {timeframe} timeframe")
                print("-" * 50)
                
                # Initialize and train strategy
                strategy = HW5_ML_Strategy(symbol=symbol, timeframe=timeframe)
                
                print(f"Training model from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
                metrics = strategy.train(start_date, end_date)
                
                print("\nModel Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric.capitalize()}: {value:.4f}")
                
                # Run backtest
                print("\nRunning backtest...")
                backtest_results = strategy.backtest(start_date, end_date, initial_balance=initial_balance)
                
                # Analyze trades
                trade_analysis = analyze_trades(backtest_results['trades'])
            
                # Store results
                results.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'metrics': metrics,
                    'backtest': backtest_results,
                    'trade_analysis': trade_analysis
                })
                
                print("\nBacktest Results:")
                print(f"Final Balance: ${backtest_results['final_balance']:,.2f}")
                print(f"Return: {backtest_results['return']:.2f}%")
                print(f"Total Trades: {trade_analysis['total_trades']}")
                print(f"Win Rate: {trade_analysis['win_rate']:.2f}%")
                print(f"Average Profit per Trade: ${trade_analysis['average_profit']:.2f}")
                
            except Exception as e:
                print(f"\nError testing {symbol} on {timeframe}:")
                print(str(e))
                traceback.print_exc()
                continue
    
    if results:
        # Compare strategies
        print("\n" + "=" * 50)
        print("Strategy Comparison Summary")
        print("=" * 50)
        
        # Sort results by return
        results.sort(key=lambda x: x['backtest']['return'], reverse=True)
        
        for result in results:
            print(f"\n{result['symbol']} - {result['timeframe']}:")
            print(f"Model Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"Return: {result['backtest']['return']:.2f}%")
            print(f"Final Balance: ${result['backtest']['final_balance']:,.2f}")
            print(f"Win Rate: {result['trade_analysis']['win_rate']:.2f}%")
            print(f"Total Trades: {result['trade_analysis']['total_trades']}")
    else:
        print("\nNo successful strategy tests completed.")

if __name__ == "__main__":
    main()
