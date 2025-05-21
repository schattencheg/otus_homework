import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ta
import streamlit as st
from datetime import datetime, timedelta
from assets.DataProvider import DataProvider
import torch
import torch.nn as nn
from assets.tsmixer import TSMixer

class HW5_ML_Strategy:
    def __init__(self, symbol='BTC/USDT', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = 24  # Look back period for time series
        # Initialize data provider with dashboard disabled to avoid errors
        self.data_provider = DataProvider(tickers=[symbol], skip_dashboard=True)
        try:
            self.data_provider.data_request()  # This loads and processes the data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        self.metrics = {}
        
        # Initialize TSMixer model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        
        # Scale features
        features = ['rsi', 'macd', 'bb_high', 'bb_low', 'open', 'high', 'low', 'close', 'volume']
        df[features] = self.scaler.fit_transform(df[features])
        
        return df
    
    def create_sequences(self, data):
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:(i + self.sequence_length)]
            target = data.iloc[i + self.sequence_length]['target']
            sequences.append(seq)
            targets.append(target)
            
        return torch.FloatTensor(np.array(sequences)), torch.LongTensor(targets)
    
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
        
        # Create sequences for TSMixer
        X_seq, y_seq = self.create_sequences(df)
        
        # Split data
        train_size = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # Initialize TSMixer model
        self.model = TSMixer(
            input_size=len(features),
            hidden_size=64,
            num_layers=2,
            seq_len=self.sequence_length,
            num_classes=2
        ).to(self.device)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 50
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.cpu().numpy()
            y_true = y_val.cpu().numpy()
            
            self.metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
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
        
        # Create sequences for prediction
        predictions = []
        for i in range(self.sequence_length, len(df)):
            seq = df[features].iloc[i-self.sequence_length:i].values
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(seq_tensor)
                _, predicted = torch.max(output.data, 1)
                predictions.append(predicted.item())
        
        # Add predictions to dataframe
        df = df.iloc[self.sequence_length:].copy()
        df['prediction'] = predictions
        
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
