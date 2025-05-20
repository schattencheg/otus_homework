import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import ta
import streamlit as st
from datetime import datetime, timedelta
from assets.DataProvider import DataProvider

class HW5_ML_Strategy:
    def __init__(self, symbol='BTC/USDT', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        # Initialize data provider with dashboard disabled to avoid errors
        self.data_provider = DataProvider(tickers=[symbol], skip_dashboard=True)
        try:
            self.data_provider.data_request()  # This loads and processes the data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        self.metrics = {}
    
    def prepare_features(self, df):
        # Add technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        
        # Create target variable (1 for price increase, 0 for decrease)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def train(self, start_date, end_date):
        try:
            # Load data
            df = self.data_provider.data.get(self.symbol)
            if df is None:
                raise ValueError(f"No data available for {self.symbol}")
            
            # Convert dates to datetime if they're not already
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filter data by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if df.empty:
                raise ValueError(f"No data available for the specified date range {start_date} to {end_date}")
            
            print(f"Training on {len(df)} data points from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
            
        # Ensure column names are lowercase
        df.columns = [x.lower() for x in df.columns]
        df = self.prepare_features(df)
        
        # Prepare features and target
        features = ['rsi', 'macd', 'bb_high', 'bb_low', 'open', 'high', 'low', 'close', 'volume']
        X = df[features]
        y = df['target']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model
        self.model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_val)
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        return self.metrics
    
    def backtest(self, start_date, end_date, initial_balance=10000):
        try:
            if self.model is None:
                raise ValueError("Model must be trained before backtesting")
                
            # Load data
            df = self.data_provider.data.get(self.symbol)
            if df is None:
                raise ValueError(f"No data available for {self.symbol}")
                
            # Convert dates to datetime if they're not already
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filter data by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if df.empty:
                raise ValueError(f"No data available for the specified date range {start_date} to {end_date}")
                
            print(f"Backtesting on {len(df)} data points from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Error during backtesting: {str(e)}")
            raise
            
        # Ensure column names are lowercase
        df.columns = [x.lower() for x in df.columns]
        df = self.prepare_features(df)
        
        features = ['rsi', 'macd', 'bb_high', 'bb_low', 'open', 'high', 'low', 'close', 'volume']
        X = df[features]
        
        # Generate predictions
        df['prediction'] = self.model.predict(X)
        
        # Simulate trading
        balance = initial_balance
        position = 0
        trades = []
        
        for i in range(len(df)-1):
            if df['prediction'].iloc[i] == 1 and position == 0:  # Buy signal
                position = balance / df['close'].iloc[i]
                balance = 0
                trades.append({
                    'type': 'buy',
                    'price': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })
            elif df['prediction'].iloc[i] == 0 and position > 0:  # Sell signal
                balance = position * df['close'].iloc[i]
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': df['close'].iloc[i],
                    'timestamp': df.index[i]
                })
        
        # Calculate final balance
        if position > 0:
            balance = position * df['close'].iloc[-1]
        
        return {
            'final_balance': balance,
            'return': (balance - initial_balance) / initial_balance * 100,
            'trades': trades
        }

def create_dashboard():
    st.title('Trading Strategy Dashboard')
    
    # Sidebar for parameters
    st.sidebar.header('Parameters')
    symbol = st.sidebar.text_input('Symbol', 'BTC/USDT')
    timeframe = st.sidebar.selectbox('Timeframe', ['1h', '4h', '1d'])
    
    # Date inputs with default values
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)
    
    start_date = st.sidebar.date_input('Start Date', value=default_start_date)
    end_date = st.sidebar.date_input('End Date', value=default_end_date, max_value=default_end_date)
    
    if start_date >= end_date:
        st.error('Error: Start date must be before end date')
    
    if st.sidebar.button('Run Strategy'):
        strategy = HW5_ML_Strategy(symbol=symbol, timeframe=timeframe)
        
        # Train and show metrics
        with st.spinner('Training model...'):
            metrics = strategy.train(start_date, end_date)
            
        st.header('Model Metrics')
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        st.table(metrics_df)
        
        # Run backtest
        with st.spinner('Running backtest...'):
            backtest_results = strategy.backtest(start_date, end_date)
        
        st.header('Backtest Results')
        st.metric('Final Balance', f"${backtest_results['final_balance']:.2f}")
        st.metric('Return', f"{backtest_results['return']:.2f}%")
        
        # Show trades
        st.header('Trade History')
        trades_df = pd.DataFrame(backtest_results['trades'])
        if not trades_df.empty:
            st.dataframe(trades_df)

if __name__ == '__main__':
    create_dashboard()
