from backtesting import Backtest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from typing import Dict, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from heapq import nlargest


class HW2Backtest:
    def __init__(self, data: pd.DataFrame):
        if data is None or data.empty:
            raise ValueError("Empty data")

        self.data = data
        self.data.columns = [x.title() for x in self.data.columns]
        self.train_data, self.validation_data = train_test_split(data, test_size=0.3, shuffle=False)

    def backtest_strategy(self, data: pd.DataFrame, strategy_class, params: Dict[str, Any]) -> Dict[str, float]:
        bt = Backtest(data, strategy_class, cash=10000000, commission=.002)
        stats = bt.run(**params)
        return stats

    def get_best_strategy(self, strategy_class, param_grid, count: int = 5) -> Dict[str, Any]:
        # Generate all combinations of parameters
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        overall_results = {}
        overall_score = {}
        overall_params = {}
        current_iteration = 0

        best_params = None
        best_score = float('-inf')
        best_scores = None
        
        # Test each combination on training data
        for i, params in tqdm(enumerate(param_combinations)):
            # Ensure sma_short is less than sma_long
            if params['sma_short'] >= params['sma_long']:
                continue
                
            # Ensure rsi_lower is less than rsi_upper if RSI parameters are present
            if 'rsi_lower' in params and 'rsi_upper' in params:
                if params['rsi_lower'] >= params['rsi_upper']:
                    continue
            
            try:
                # Run backtest with current parameters
                stats = self.backtest_strategy(self.train_data, strategy_class, params)
                stats['Trades'] = len(stats['_trades'])

                # Calculate score based on multiple metrics
                # We want high returns, high Sharpe ratio, and low drawdown
                score = (stats['Return [%]'] * 0.4 +  # 40% weight on returns
                         stats['Sharpe Ratio'] * 0.3 +  # 30% weight on risk-adjusted returns
                         -stats['Max. Drawdown [%]'] * 0.2 +  # 20% weight on drawdown (negative because we want to minimize)
                         stats['Win Rate [%]'] * 0.1)  # 10% weight on win rate
                
                overall_score[current_iteration] = score
                overall_params[current_iteration] = params.copy()
                overall_results[current_iteration] = stats
                current_iteration += 1
                    
                if (i + 1) % 10 == 0:
                    best_score = max(overall_score.values())
                    #tqdm.write(f"Tested {i + 1}/{len(param_combinations)} combinations. Best score so far: {best_score:.2f}")
                    
            except Exception as e:
                print(f"Error testing parameters {params}: {str(e)}")
                continue
        
        if not overall_score:
            print("\nNo valid parameter combinations found. All combinations resulted in errors.")
            return None
            
        idxs = [x[0] for x in nlargest(count, overall_score.items(), key=lambda i: i[1])]
        best_params = [overall_params[i] for i in idxs]
        best_scores = [overall_score[i] for i in idxs]
        best_results = [overall_results[i] for i in idxs]
        print("\nBest parameters found:")
        for params, score in zip(best_params, best_scores):
            print(f"{score:.2f}: {params}")
        best_score = best_scores[0]
        best_result = best_results[0]

        print(f"\nBest score: {best_score:.2f}")
        print("\nBest strategy performance:")
        print(f"Return: {best_result['Return [%]']:.2f}%")
        print(f"Trades count: {best_result['Trades']}")
        print(f"Sharpe Ratio: {best_result['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {best_result['Max. Drawdown [%]']:.2f}%")
        print(f"Win Rate: {best_result['Win Rate [%]']:.2f}%")
            
        return best_params

    def create_performance_dashboard(self, strategies_results: Dict[str, Dict[str, float]]) -> None:
        """Create a dashboard comparing different strategies performance"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Comparison', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate')
        )

        strategies = list(strategies_results.keys())
        returns = [results['Return [%]'] for results in strategies_results.values()]
        sharpe = [results['Sharpe Ratio'] for results in strategies_results.values()]
        drawdowns = [results['Max. Drawdown [%]'] for results in strategies_results.values()]
        win_rates = [results['Win Rate [%]'] for results in strategies_results.values()]

        # Returns
        fig.add_trace(go.Bar(x=strategies, y=returns, name='Return %'), row=1, col=1)
        
        # Sharpe Ratio
        fig.add_trace(go.Bar(x=strategies, y=sharpe, name='Sharpe'), row=1, col=2)
        
        # Max Drawdown
        fig.add_trace(go.Bar(x=strategies, y=drawdowns, name='Drawdown %'), row=2, col=1)
        
        # Win Rate
        fig.add_trace(go.Bar(x=strategies, y=win_rates, name='Win Rate %'), row=2, col=2)

        fig.update_layout(height=800, width=1200, title_text='Strategy Performance Comparison')
        fig.show()


if __name__ == "__main__":
    pass