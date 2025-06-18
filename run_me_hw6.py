import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import streamlit as st
from prophet import Prophet
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stable_baselines3 import DQN, PPO, A2C
from assets.hw6.SimpleBacktester import SimpleBacktester
from assets.DataProvider import DataProvider
from assets.enums import DataResolution, DataPeriod
from assets.FeaturesGenerator import FeaturesGenerator
import ta
import gymnasium as gym
from gymnasium import spaces


# Model architectures
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class CNNModel(nn.Module):
    def __init__(self, input_channels, seq_length):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TSMixerModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super().__init__()
        self.temporal_mix = nn.Linear(input_size, hidden_size)
        self.feature_mix = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        temp_mix = self.temporal_mix(x)
        feat_mix = self.feature_mix(temp_mix)
        return self.output(feat_mix)

def main():
    # Generate sample data for testing
    print("Generating sample data...")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic price data with some realistic patterns
    base_price = 30000
    trend = np.linspace(0, 5000, len(dates))  # Upward trend
    noise = np.random.normal(0, 500, len(dates))  # Random noise
    seasonal = 1000 * np.sin(np.linspace(0, 10*np.pi, len(dates)))  # Seasonal pattern
    
    close_prices = base_price + trend + noise + seasonal
    high_prices = close_prices + np.random.uniform(0, 200, len(dates))
    low_prices = close_prices - np.random.uniform(0, 200, len(dates))
    open_prices = close_prices + np.random.uniform(-100, 100, len(dates))
    volumes = np.random.uniform(100, 1000, len(dates)) * (1 + 0.5 * np.sin(np.linspace(0, 8*np.pi, len(dates))))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    print(f"Generated {len(data)} data points")
    
    # Prepare features
    features = prepare_features(data)
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = features.shape[1]
    seq_length = 60
    
    lstm_model = LSTMModel(input_dim).to(device)
    cnn_model = CNNModel(input_dim, seq_length).to(device)
    tsmixer_model = TSMixerModel(input_dim, seq_length).to(device)
    
    # Initialize Prophet model
    prophet_model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative'
    )
    
    # Create custom gym environment for RL agents
    import gymnasium as gym
    from gymnasium import spaces
    
    class TradingEnv(gym.Env):
        def __init__(self, data, features):
            super().__init__()
            self.data = data
            self.features = features
            self.current_step = 0
            
            # Define action and observation spaces
            self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32
            )
        
        def reset(self, seed=None):
            super().reset(seed=seed)
            self.current_step = 0
            return self.features.iloc[self.current_step].values, {}
        
        def step(self, action):
            # Execute action and get reward
            reward = self._get_reward(action)
            
            # Move to next step
            self.current_step += 1
            done = self.current_step >= len(self.features) - 1
            
            # Get next observation
            if not done:
                next_obs = self.features.iloc[self.current_step].values
            else:
                next_obs = self.features.iloc[-1].values
            
            return next_obs, reward, done, False, {}
        
        def _get_reward(self, action):
            current_price = self.data['Close'].iloc[self.current_step]
            next_price = self.data['Close'].iloc[min(self.current_step + 1, len(self.data) - 1)]
            price_change = (next_price - current_price) / current_price
            
            # Calculate volatility penalty
            volatility = self.features['returns_vol'].iloc[self.current_step]
            vol_penalty = -abs(volatility) * 0.1
            
            # Calculate trend alignment bonus
            trend = self.features['momentum_20'].iloc[self.current_step]
            trend_bonus = np.sign(trend) * abs(trend) * 0.05
            
            # Base reward
            if action == 0:  # Hold
                base_reward = 0
            elif action == 1:  # Buy
                base_reward = price_change
            else:  # Sell
                base_reward = -price_change
            
            # Add risk-adjusted components
            total_reward = base_reward + vol_penalty + trend_bonus
            
            # Apply position sizing based on confidence
            confidence = abs(self.features['adx'].iloc[self.current_step]) / 100
            total_reward *= confidence
            
            return total_reward
    
    # Create environment
    env = TradingEnv(data, features)
    
    # Initialize RL agents with the environment
    print("Initializing RL agents...")
    dqn_agent = DQN('MlpPolicy', env=env, verbose=1)
    ppo_agent = PPO('MlpPolicy', env=env, verbose=1)
    a2c_agent = A2C('MlpPolicy', env=env, verbose=1)
    
    # Train agents (just a few steps for demonstration)
    print("Training RL agents...")
    dqn_agent.learn(total_timesteps=1000)
    ppo_agent.learn(total_timesteps=1000)
    a2c_agent.learn(total_timesteps=1000)
    
    # Initialize sentiment analyzer
    print("Initializing sentiment analyzer...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    
    # Initialize backtester
    backtester = SimpleBacktester(
        data=data,
        features=features,
        lstm_model=lstm_model,
        cnn_model=cnn_model,
        voting_model=tsmixer_model
    )
    
    # Run backtest
    results = backtester.run()
    
    # Launch dashboard
    launch_dashboard(results, backtester)

def prepare_features(df):
    features = pd.DataFrame(index=df.index)
    
    features_generator = FeaturesGenerator()
    ohlc_for_features = df[['Open', 'High', 'Low', 'Close']]
    all_features_df, _ = features_generator.prepare_features(ohlc_for_features)
    
    # Price action features
    features['returns'] = df['Close'].pct_change()
    features['returns_vol'] = features['returns'].rolling(window=20).std()
    vol_quantiles = features['returns_vol'].quantile([0.33, 0.66])
    features['volatility_regime'] = pd.cut(features['returns_vol'],
                                            bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.66], np.inf],
                                            labels=[0, 1, 2])  # 0: low, 1: medium, 2: high
    features['body_size'] = (df['Close'] - df['Open']) / df['Open']
    features['upper_shadow'] = (df['High'] - df['Close']) / df['Open']
    features['lower_shadow'] = (df['Open'] - df['Low']) / df['Open']
    features['high_low_range'] = df['High'] - df['Low']

    # Calculate moving averages
    for window in [5, 10, 20, 40]:
        features[f'sma_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
        features[f'ema_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
        
        # Price distance from MA
        features[f'ma_dist_{window}'] = (df['Close'] - features[f'sma_{window}']) / features[f'sma_{window}']
    
    # RSI
    features['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
    features['rsi_14_slope'] = features['rsi_14'].diff()
    features['rsi_14_ma'] = features['rsi_14'].rolling(window=5).mean()
    
    # MACD
    # MACD with different parameters
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    features['macd'] = macd.macd()
    features['macd_signal'] = macd.macd_signal()
    # Calculate MACD histogram as difference between MACD and signal line
    features['macd_hist'] = features['macd'] - features['macd_signal']
    features['macd_hist_slope'] = features['macd_hist'].diff()
    
    # Bollinger Bands
    for window in [20, 40]:
        bb = ta.volatility.BollingerBands(df['Close'], window=window)
        features[f'bb_position_{window}'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        features[f'bb_width_{window}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # Volatility indicators
    features['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    features['natr'] = features['atr'] / df['Close']
    
    # Volume analysis
    features['volume_sma'] = df['Volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['Volume'] / features['volume_sma']
    features['volume_price_trend'] = features['volume_ratio'] * features['returns'].apply(np.sign)
    features['volume_price_corr'] = df['Volume'].rolling(window=20).corr(df['Close'])

    # Trend strength
    features['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    features['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
    features['trend_strength'] = features['adx'] / 100  # Normalized trend strength
    features['trend_direction'] = np.where(features['macd'] > features['macd_signal'], 1, -1)

    # Volume profile analysis
    features['volume_force'] = df['Volume'] * features['returns']
    features['money_flow_index'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Price patterns
    features['high_low_ratio'] = df['High'] / df['Low']
    features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Momentum
    for window in [10, 20, 30]:
        features[f'momentum_{window}'] = df['Close'].diff(periods=window) / df['Close'].shift(window)
        features[f'roc_{window}'] = ta.momentum.roc(df['Close'], window=window)
    
    # Fill NaN values with forward fill then backward fill
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Normalize features
    for col in features.columns:
        if col != 'returns' and col != 'volatility_regime':
            features[col] = (features[col] - features[col].mean()) / features[col].std()
    
    return features

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices, period=20, std=2):
    middle = prices.rolling(period).mean()
    std_dev = prices.rolling(period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return upper, middle, lower

def launch_dashboard(results, backtester):
    st.title('Trading Strategy Dashboard')
    
    # Performance metrics
    st.header('Performance Metrics')
    total_return = ((backtester.cash / backtester.initial_cash) - 1) * 100
    win_rate = (backtester.winning_trades / backtester.total_trades * 100 
                if backtester.total_trades > 0 else 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Return', f'{total_return:.2f}%')
    col2.metric('Win Rate', f'{win_rate:.2f}%')
    col3.metric('Max Drawdown', f'{backtester.max_drawdown:.2f}%')
    
    # Trade history
    st.header('Trade History')
    st.dataframe(results)
    
    # Equity curve
    st.header('Equity Curve')
    equity_curve = pd.DataFrame({
        'Date': results['entry_time'],
        'Equity': backtester.cash + results['pnl'].cumsum()
    })
    st.line_chart(equity_curve.set_index('Date'))

if __name__ == '__main__':
    main()
