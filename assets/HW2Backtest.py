from backtesting import Backtest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from typing import Dict, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class HW2Backtest:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data.columns = [x.title() for x in self.data.columns]
        self.train_data, self.test_val_data = train_test_split(data, test_size=0.3, shuffle=False)
        self.test_data, self.validation_data = train_test_split(self.test_val_data, test_size=0.5, shuffle=False)

    def bactest_strategy(self, data: pd.DataFrame, strategy_class, params: Dict[str, Any]) -> Dict[str, float]:
        bt = Backtest(data, strategy_class, cash=10000000, commission=.002)
        stats = bt.run(**params)
        return stats

    def get_best_strategy(self, strategy_class) -> Dict[str, Any]:
        # Define default parameters
        default_params = {
            'sma_short': 10,
            'sma_long': 20,
            'rsi_period': 14,
            'rsi_upper': 70,
            'rsi_lower': 30
        }
        return default_params


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