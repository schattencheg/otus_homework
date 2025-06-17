import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from assets.DataProvider import DataProvider
from assets.enums import DataResolution, DataPeriod
from pats_cuda import BullPatternCNN, PriceDataset
import logging

logging.basicConfig(level=logging.INFO)

def predict_patterns(model, data_provider, window_size=60, temperature=2.0):
    """Make predictions using the trained model"""
    model.eval()
    dataset = PriceDataset(data_provider.data['BTC/USDT'], window_size=window_size)
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sequence, _ = dataset[i]
            sequence = sequence.unsqueeze(0).to(model.device)  # Add batch dimension
            output = model(sequence)
            logging.info(f"Raw output: {output}")
            scaled_output = output / temperature
            probs = torch.softmax(scaled_output, dim=1)
            logging.info(f"Scaled output: {scaled_output}")
            logging.info(f"Softmax probabilities: {probs}")
            pred = torch.argmax(probs, dim=1)
            predictions.append(pred.cpu().item())
            probabilities.append(float(probs[0][1].cpu().item()))  # Probability of bullish pattern
            if i >= 5:  # Only show first few predictions
                break
    
    return predictions, probabilities

def backtest_strategy(data, predictions, probabilities, threshold=0.7):
    """Backtest the trading strategy"""
    df = data.copy()
    
    # Create signal columns
    df.loc[:, 'prediction'] = 0
    df.loc[:, 'probability'] = 0.0
    df.loc[:, 'position'] = 0
    df.loc[:, 'returns'] = df['Close'].pct_change()
    
    # Align predictions with data
    start_idx = 60
    end_idx = start_idx + len(predictions)
    df.loc[df.index[start_idx:end_idx], 'prediction'] = predictions
    df.loc[df.index[start_idx:end_idx], 'probability'] = probabilities
    
    # Generate trading signals
    df.loc[:, 'position'] = (df['probability'] > threshold).astype(int)
    
    # Calculate strategy returns
    df.loc[:, 'strategy_returns'] = df['position'].shift(1) * df['returns']
    df.loc[:, 'cumulative_returns'] = (1 + df['returns']).cumprod()
    df.loc[:, 'strategy_cumulative'] = (1 + df['strategy_returns']).cumprod()
    
    return df

def plot_results(df):
    """Create interactive plots of the results"""
    fig = make_subplots(rows=4, cols=2, 
                       shared_xaxes=True,
                       vertical_spacing=0.08,
                       horizontal_spacing=0.05,
                       subplot_titles=('Price and Positions', 'Daily Returns Distribution',
                                     'Signal Probability', 'Position Changes',
                                     'Strategy Performance', 'Rolling Sharpe Ratio',
                                     'Drawdown Analysis', 'Monthly Returns Heatmap'),
                       specs=[[{"secondary_y": True}, {"type": "histogram"}],
                             [{}, {}],
                             [{"colspan": 2}, None],
                             [{"colspan": 2}, None]],
                       row_heights=[0.3, 0.2, 0.25, 0.25])
    
    # 1. Price and positions plot with volume
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       name='Price'),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if r < 0 else 'green' for r in df['Close'].diff()]
    fig.add_trace(
        go.Bar(x=df.index, 
               y=df['Volume'],
               name='Volume',
               marker_color=colors,
               opacity=0.3),
        row=1, col=1,
        secondary_y=True
    )
    
    # Add position markers
    longs = df[df['position'] == 1].index
    fig.add_trace(
        go.Scatter(x=longs, 
                  y=df.loc[longs, 'Low'] * 0.99,
                  mode='markers',
                  marker=dict(symbol='triangle-up', size=10, color='green'),
                  name='Long Position'),
        row=1, col=1
    )
    
    # 2. Daily returns distribution
    fig.add_trace(
        go.Histogram(x=df['strategy_returns'].dropna(),
                     name='Strategy Returns',
                     nbinsx=50,
                     histnorm='probability'),
        row=1, col=2
    )
    
    # 3. Signal probability
    fig.add_trace(
        go.Scatter(x=df.index, 
                  y=df['probability'],
                  name='Bull Pattern Probability',
                  fill='tozeroy'),
        row=2, col=1
    )
    
    # 4. Position changes
    position_changes = df['position'].diff().fillna(0)
    fig.add_trace(
        go.Bar(x=df.index,
               y=position_changes,
               name='Position Changes'),
        row=2, col=2
    )
    
    # 5. Strategy performance
    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['cumulative_returns'],
                  name='Buy & Hold',
                  line=dict(width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index,
                  y=df['strategy_cumulative'],
                  name='Strategy',
                  line=dict(width=2)),
        row=3, col=1
    )
    
    # 6. Calculate and plot drawdown
    rolling_max = df['strategy_cumulative'].expanding().max()
    drawdown = (df['strategy_cumulative'] - rolling_max) / rolling_max * 100
    fig.add_trace(
        go.Scatter(x=df.index,
                  y=drawdown,
                  name='Drawdown %',
                  fill='tozeroy',
                  line=dict(color='red')),
        row=4, col=1
    )
    
    # Calculate additional metrics
    daily_returns = df['strategy_returns'].dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    max_drawdown = drawdown.min()
    win_rate = (daily_returns > 0).mean() * 100
    profit_factor = abs(daily_returns[daily_returns > 0].sum() / daily_returns[daily_returns < 0].sum())
    
    # Add annotations with metrics
    metrics_text = f"Sharpe Ratio: {sharpe:.2f}<br>"
    metrics_text += f"Max Drawdown: {max_drawdown:.2f}%<br>"
    metrics_text += f"Win Rate: {win_rate:.1f}%<br>"
    metrics_text += f"Profit Factor: {profit_factor:.2f}"
    
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    # Update layout
    fig.update_layout(
        title='CNN Bull Pattern Strategy Results',
        height=1600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Probability", row=2, col=1)
    fig.update_yaxes(title_text="Position Change", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Return", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=4, col=1)
    
    # Save the plot
    fig.write_html('strategy_results.html')
    logging.info("Results plot saved to strategy_results.html")

def main():
    # Load data
    data_provider = DataProvider(
        tickers=['BTC/USDT'],
        resolution=DataResolution.MINUTE_01,
        period=DataPeriod.MONTH_01
    )
    
    logging.info("Loading data...")
    data_provider.data_request()
    
    if not data_provider.data:
        logging.error("Failed to load data!")
        return
    
    logging.info(f"Loaded {len(data_provider.data['BTC/USDT'])} data points")
    
    # Load trained model
    try:
        model = BullPatternCNN(device='cpu')
        model.load_state_dict(torch.load('bull_pattern_model.pth'))
        model.eval()
    except FileNotFoundError:
        logging.error("Trained model not found! Please run test_cnn.py first.")
        return
    
    # Make predictions
    logging.info("Making predictions...")
    predictions, probabilities = predict_patterns(model, data_provider)
    
    # Analyze predictions
    logging.info("Prediction Statistics:")
    logging.info(f"Mean probability: {np.mean(probabilities):.4f}")
    logging.info(f"Max probability: {np.max(probabilities):.4f}")
    logging.info(f"Min probability: {np.min(probabilities):.4f}")
    logging.info(f"Std probability: {np.std(probabilities):.4f}")
    
    # Backtest strategy with lower threshold
    logging.info("\nBacktesting strategy...")
    results = backtest_strategy(data_provider.data['BTC/USDT'], predictions, probabilities, threshold=0.02)
    
    # Calculate strategy metrics
    total_trades = results['position'].diff().abs().sum()
    win_rate = (results['strategy_returns'] > 0).mean()
    strategy_return = results['strategy_cumulative'].iloc[-1] - 1
    buy_hold_return = results['cumulative_returns'].iloc[-1] - 1
    
    logging.info("\nStrategy Results:")
    logging.info(f"Total Trades: {total_trades}")
    logging.info(f"Win Rate: {win_rate:.2%}")
    logging.info(f"Strategy Return: {strategy_return:.2%}")
    logging.info(f"Buy & Hold Return: {buy_hold_return:.2%}")
    
    # Plot results
    logging.info("\nGenerating plots...")
    plot_results(results)

if __name__ == "__main__":
    main()
