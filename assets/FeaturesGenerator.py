from typing import Tuple, List
import numpy as np
import pandas as pd
import logging

class FeaturesGenerator:
    def __init__(self):
        pass
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix X and list of feature names with enhanced technical indicators"""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")

        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        try:
            # First calculate returns and basic metrics
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Calculate Stochastic Oscillator
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            
            # Create momentum indicators
            for period in [5, 10, 20]:
                df[f'ROC_{period}'] = df['Close'].pct_change(period)
                df[f'MOM_{period}'] = df['Close'].diff(period)
            
            # Calculate ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # Calculate volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Returns_MA'] = df['Returns'].rolling(window=20).mean()
            
            # Volume-based features
            if 'Volume' in df.columns:
                df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Std'] = df['Volume'].rolling(window=20).std()
                df['Volume_ZScore'] = (df['Volume'] - df['Volume_MA']) / df['Volume_Std']
                df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # Price-based features
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

            # Additional OHLC-based features
            # Candlestick features
            df['Body_Size'] = np.abs(df['Open'] - df['Close'])
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Is_Bullish_Candle'] = (df['Close'] > df['Open']).astype(int)
            df['Is_Bearish_Candle'] = (df['Close'] < df['Open']).astype(int)

            # Price vs MAs (ensure MAs are calculated first)
            if 'SMA_50' in df.columns: # SMA_50 is calculated above
                df['Close_vs_SMA_50'] = df['Close'] - df['SMA_50']
            if 'EMA_20' in df.columns: # EMA_20 is calculated above
                df['Close_vs_EMA_20'] = df['Close'] - df['EMA_20']

            # Rolling Min/Max
            rolling_window_new = 10
            df[f'Rolling_Max_Close_{rolling_window_new}'] = df['Close'].rolling(window=rolling_window_new).max()
            df[f'Rolling_Min_Close_{rolling_window_new}'] = df['Close'].rolling(window=rolling_window_new).min()
            # Avoid division by zero or NaN by replacing 0 with a very small number or handling after
            df[f'Close_vs_Rolling_Max_{rolling_window_new}'] = df['Close'] / df[f'Rolling_Max_Close_{rolling_window_new}'].replace(0, np.nan)
            df[f'Close_vs_Rolling_Min_{rolling_window_new}'] = df['Close'] / df[f'Rolling_Min_Close_{rolling_window_new}'].replace(0, np.nan)
            
            # Gap feature
            df['Open_vs_Prev_Close'] = df['Open'] - df['Close'].shift(1)

            # Time-based Features (assuming df.index is DatetimeIndex)
            if isinstance(df.index, pd.DatetimeIndex):
                df['Day_Of_Week'] = df.index.dayofweek
                df['Month_Of_Year'] = df.index.month
                df['Week_Of_Year'] = df.index.isocalendar().week.astype(int)
                df['Quarter'] = df.index.quarter

            # Lagged Features
            for lag in [1, 2, 3]:
                df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df['Log_Returns_Lag_1'] = df['Log_Returns'].shift(1)
            if 'Volatility' in df.columns:
                 df['Volatility_Lag_1'] = df['Volatility'].shift(1)

            # Volatility Variations
            if 'Daily_Range' in df.columns:
                df['Daily_Range_MA_10'] = df['Daily_Range'].rolling(window=10).mean()
                df['Daily_Range_Std_10'] = df['Daily_Range'].rolling(window=10).std()
                df['Daily_Range_vs_MA_10'] = df['Daily_Range'] / df['Daily_Range_MA_10'].replace(0, np.nan)
            if 'ATR' in df.columns:
                df['ATR_Normalized'] = df['ATR'] / df['Close'].replace(0, np.nan)

            # Trend/Momentum Variations
            if 'SMA_20' in df.columns:
                df['SMA_20_Slope'] = df['SMA_20'].diff()
            if 'EMA_10' in df.columns:
                df['EMA_10_Slope'] = df['EMA_10'].diff()
            if 'SMA_5' in df.columns and 'SMA_20' in df.columns:
                df['Price_Oscillator_5_20'] = df['SMA_5'] - df['SMA_20']

            # Interaction Features
            if 'RSI' in df.columns and 'Volume_ZScore' in df.columns and 'Volume' in df.columns:
                df['RSI_x_Volume_ZScore'] = df['RSI'] * df['Volume_ZScore']
            if 'MACD' in df.columns and 'RSI' in df.columns:
                df['MACD_x_RSI'] = df['MACD'] * df['RSI']

            # Drop NaN values introduced by rolling windows, shifts, and calculations
            df = df.dropna(axis=1, how='all').dropna()

            # Define feature columns
            feature_columns = [
                'RSI', 'MACD', 'MACD_Signal',
                'BB_Lower', 'BB_Middle', 'BB_Upper',
                'Stoch_K', 'Stoch_D',
                'ATR', 'Daily_Range', 'Volatility',
                'Log_Returns'
            ]
            
            # Add momentum indicators
            feature_columns.extend([f'ROC_{p}' for p in [5, 10, 20]])
            feature_columns.extend([f'MOM_{p}' for p in [5, 10, 20]])
            
            # Add moving averages
            feature_columns.extend([f'SMA_{p}' for p in [5, 10, 20, 50, 200]])
            feature_columns.extend([f'EMA_{p}' for p in [5, 10, 20, 50, 200]])
            
            # Add volume features if available
            if 'Volume' in df.columns:
                feature_columns.extend(['Volume_MA', 'Volume_Std', 'Volume_ZScore', 'OBV'])

            # Add new OHLC-based features to the list
            new_feature_names = [
                'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Is_Bullish_Candle', 'Is_Bearish_Candle',
                'Open_vs_Prev_Close'
            ]
            if 'SMA_50' in df.columns: # Add only if base MA was calculated and column exists
                new_feature_names.append('Close_vs_SMA_50')
            if 'EMA_20' in df.columns:
                new_feature_names.append('Close_vs_EMA_20')
            
            rolling_window_new = 10 # ensure this matches the window used in calculation
            new_feature_names.extend([
                f'Rolling_Max_Close_{rolling_window_new}', f'Rolling_Min_Close_{rolling_window_new}',
                f'Close_vs_Rolling_Max_{rolling_window_new}', f'Close_vs_Rolling_Min_{rolling_window_new}'
            ])
            feature_columns.extend(new_feature_names)

            # Add even more trading features to the list
            additional_trading_features = []
            if isinstance(df.index, pd.DatetimeIndex):
                additional_trading_features.extend(['Day_Of_Week', 'Month_Of_Year', 'Week_Of_Year', 'Quarter'])
            
            for lag in [1, 2, 3]:
                additional_trading_features.append(f'Returns_Lag_{lag}')
            additional_trading_features.append('Log_Returns_Lag_1')
            if 'Volatility' in df.columns: # Check if base feature exists
                 additional_trading_features.append('Volatility_Lag_1')

            if 'Daily_Range' in df.columns:
                additional_trading_features.extend(['Daily_Range_MA_10', 'Daily_Range_Std_10', 'Daily_Range_vs_MA_10'])
            if 'ATR' in df.columns:
                additional_trading_features.append('ATR_Normalized')

            if 'SMA_20' in df.columns:
                additional_trading_features.append('SMA_20_Slope')
            if 'EMA_10' in df.columns:
                additional_trading_features.append('EMA_10_Slope')
            if 'SMA_5' in df.columns and 'SMA_20' in df.columns:
                additional_trading_features.append('Price_Oscillator_5_20')

            if 'RSI' in df.columns and 'Volume_ZScore' in df.columns and 'Volume' in df.columns:
                additional_trading_features.append('RSI_x_Volume_ZScore')
            if 'MACD' in df.columns and 'RSI' in df.columns:
                additional_trading_features.append('MACD_x_RSI')
            
            feature_columns.extend(additional_trading_features)
            
            # Prepare feature matrix
            for column_name in feature_columns:
                if column_name not in df.columns:
                    feature_columns.remove(column_name)
            X = df[feature_columns].copy()
            
            # Handle missing and infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Remove any remaining NaN values
            valid_idx = ~X.isna().any(axis=1)
            X = X[valid_idx]
            return X.copy(), feature_columns
            
        except Exception as e:
            logging.error(f"Error in prepare_features: {str(e)}")
            raise
