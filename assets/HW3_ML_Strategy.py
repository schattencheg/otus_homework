import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from typing import Dict, Tuple, Optional
from assets.DataProvider import DataProvider

logging.basicConfig(level=logging.INFO)

class HW3_ML_Strategy:
    def __init__(self, data_provider: DataProvider, test_size: float = 0.2, random_state: int = 42,
                 n_estimators: int = 100, max_depth: int = 10, min_samples_split: int = 5):
        """
        Initialize ML-based trading strategy
        
        Args:
            data_provider: DataProvider instance with processed market data
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the random forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split node
        """
        self.data_provider: DataProvider = data_provider
        self.test_size = test_size
        self.random_state = random_state
        self.dashboard: go.Figure = None

        # Initialize model with more specific parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        self.metrics = {}
        self.is_fitted = False
        
        # Create directory for strategy results
        self.results_dir = os.path.join('results', 'ml_strategy')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target vector y with enhanced technical indicators"""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")

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
            
            # Drop NaN values before creating target variable
            df = df.dropna()
            
            # Create target variable (1: Buy, 0: Hold, -1: Sell)
            y = pd.Series(0, index=df.index)
            
            # Buy signal: return > mean + 0.5 * volatility
            buy_mask = df['Returns'] > (df['Returns_MA'] + 0.5 * df['Volatility'])
            y[buy_mask] = 1
            
            # Sell signal: return < mean - 0.5 * volatility
            sell_mask = df['Returns'] < (df['Returns_MA'] - 0.5 * df['Volatility'])
            y[sell_mask] = -1
            
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
            
            # Prepare feature matrix
            X = df[feature_columns].copy()
            
            # Handle missing and infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Remove any remaining NaN values
            valid_idx = ~X.isna().any(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) == 0:
                raise ValueError("No valid data points after preprocessing")
            
            # Scale features
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
            logging.info(f"Final dataset shape - Features: {X.shape}, Target: {y.shape}")
            logging.info(f"Class distribution - Buy: {(y == 1).sum()}, Hold: {(y == 0).sum()}, Sell: {(y == -1).sum()}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error in prepare_features: {str(e)}")
            raise
    
    def train(self, ticker: str) -> pd.DataFrame:
        """Train the ML model on historical data using time series cross-validation"""
        try:
            # Get processed data for the ticker
            df = self.data_provider.data_processed[ticker]
            if df is None or df.empty:
                raise ValueError(f"No processed data available for {ticker}")
            
            # Prepare features and target
            X, y = self.prepare_features(df)
            
            # Use TimeSeriesSplit for more realistic evaluation
            n_splits = min(5, len(X) // 30)  # Ensure we have at least 30 samples per split
            if n_splits < 2:
                raise ValueError(f"Not enough data points for cross-validation. Need at least 60, got {len(X)}")
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            all_predictions = []
            feature_importances = []
            
            for train_idx, test_idx in tscv.split(X):
                # Split data
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train model
                self.model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test_scaled)
                
                # Store predictions and feature importance
                predictions = pd.DataFrame({
                    'y_true': y_test,
                    'y_pred': y_pred
                }, index=X_test.index)
                
                all_predictions.append(predictions)
                feature_importances.append(pd.Series(self.model.feature_importances_, index=X.columns))
            
            # Combine all predictions
            final_predictions = pd.concat(all_predictions)
            
            # Average feature importance across folds
            avg_feature_importance = pd.concat(feature_importances, axis=1).mean(axis=1)
            
            # Calculate and store metrics
            self.metrics[ticker] = {
                'classification_report': classification_report(
                    final_predictions['y_true'],
                    final_predictions['y_pred'],
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(
                    final_predictions['y_true'],
                    final_predictions['y_pred']
                ),
                'feature_importance': dict(avg_feature_importance)
            }
            
            # Train final model on all data for future predictions
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            return final_predictions
            
        except Exception as e:
            logging.error(f"Error training model for {ticker}: {str(e)}")
            raise
    
    def predict(self, ticker: str, data: pd.DataFrame = None) -> pd.Series:
        """Make predictions on new data"""
        try:
            if not self.is_fitted:
                raise NotFittedError("Model must be trained before making predictions")
                
            if data is None:
                data = self.data_provider.data_processed[ticker]
                if data is None or data.empty:
                    raise ValueError(f"No data available for {ticker}")
            
            X, _ = self.prepare_features(data)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            return pd.Series(predictions, index=X.index)
            
        except Exception as e:
            logging.error(f"Error making predictions for {ticker}: {str(e)}")
            raise
    
    def dashboard_create(self, ticker: str, predictions: pd.DataFrame) -> go.Figure:
        """Create an interactive dashboard showing strategy performance and metrics"""
        try:
            df = self.data_provider.data_processed[ticker]
            if df is None or df.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Create figure with subplots
            self.dashboard = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=(
                    'Price and Signals',
                    'Volume and Indicators',
                    'Model Performance Metrics',
                    'Feature Importance'
                )
            )
            
            # 1. Price chart with buy/sell signals
            self.dashboard.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add signals
            for signal, color, symbol, offset in [
                (1, 'green', 'triangle-up', 0.99),  # Buy
                (-1, 'red', 'triangle-down', 1.01)   # Sell
            ]:
                signal_points = predictions[predictions['y_pred'] == signal].index
                if len(signal_points) > 0:
                    self.dashboard.add_trace(
                        go.Scatter(
                            x=signal_points,
                            y=df.loc[signal_points, 'Low' if signal == 1 else 'High'] * offset,
                            mode='markers',
                            marker=dict(symbol=symbol, size=10, color=color),
                            name=f"{'Buy' if signal == 1 else 'Sell'} Signal"
                        ),
                        row=1, col=1
                    )
            
            # 2. Volume and technical indicators
            if 'Volume' in df.columns:
                self.dashboard.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name='Volume',
                        marker_color='rgba(100,100,100,0.5)'
                    ),
                    row=2, col=1
                )
            
            # Add RSI
            if 'RSI_14' in df.columns:
                self.dashboard.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI_14'],
                        name='RSI'
                    ),
                    row=2, col=1
                )
            
            # 3. Model performance metrics
            if ticker in self.metrics:
                metrics = self.metrics[ticker]['classification_report']
                classes = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
                metrics_df = pd.DataFrame({
                    'Precision': [metrics[str(i)]['precision'] for i in [-1, 0, 1]],
                    'Recall': [metrics[str(i)]['recall'] for i in [-1, 0, 1]],
                    'F1-Score': [metrics[str(i)]['f1-score'] for i in [-1, 0, 1]]
                }, index=classes)
                
                for metric in ['Precision', 'Recall', 'F1-Score']:
                    self.dashboard.add_trace(
                        go.Bar(
                            x=classes,
                            y=metrics_df[metric],
                            name=metric
                        ),
                        row=3, col=1
                    )
            
            # 4. Feature importance
            if ticker in self.metrics:
                feature_imp = pd.Series(self.metrics[ticker]['feature_importance'])
                feature_imp = feature_imp.sort_values(ascending=True)
                top_n = 15  # Show only top 15 features
                
                self.dashboard.add_trace(
                    go.Bar(
                        x=feature_imp[-top_n:].values,
                        y=feature_imp[-top_n:].index,
                        orientation='h',
                        name='Feature Importance'
                    ),
                    row=4, col=1
                )
            
            # Update layout
            self.dashboard.update_layout(
                title=dict(
                    text=f'ML Strategy Dashboard - {ticker}',
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                height=1200,
                template='plotly_white'
            )
            
            # Save dashboard
            output_path = os.path.join(self.results_dir, f'{ticker.replace("/", "_")}_dashboard.html')
            self.dashboard.write_html(output_path)
            logging.info(f"Dashboard saved to {output_path}")
            
            return self.dashboard
            
        except Exception as e:
            logging.error(f"Error creating dashboard for {ticker}: {str(e)}")
            raise
    
    def dashboard_show(self):
        if self.dashboard is None:
            raise ValueError("Dashboard not created yet. Please run create_dashboard first.")
        
        self.dashboard.show()

    def save_metrics(self, ticker: str) -> None:
        """Save detailed strategy metrics to files"""
        try:
            if ticker not in self.metrics:
                logging.warning(f"No metrics available for {ticker}")
                return
            
            # Create metrics directory if it doesn't exist
            metrics_dir = os.path.join(self.results_dir, 'metrics')
            if not os.path.exists(metrics_dir):
                os.makedirs(metrics_dir)
            
            # 1. Overall metrics
            metrics = self.metrics[ticker]['classification_report']
            overall_metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1-score'],
                'Value': [
                    metrics['accuracy'],
                    metrics['macro avg']['precision'],
                    metrics['macro avg']['recall'],
                    metrics['macro avg']['f1-score']
                ]
            })
            
            # 2. Class-specific metrics
            class_metrics = []
            for label in [-1, 0, 1]:
                class_metrics.append({
                    'Class': f"{'Sell' if label == -1 else 'Hold' if label == 0 else 'Buy'} ({label})",
                    'Precision': metrics[str(label)]['precision'],
                    'Recall': metrics[str(label)]['recall'],
                    'F1-score': metrics[str(label)]['f1-score'],
                    'Support': metrics[str(label)]['support']
                })
            class_metrics_df = pd.DataFrame(class_metrics)
            
            # 3. Feature importance
            feature_imp = pd.DataFrame({
                'Feature': list(self.metrics[ticker]['feature_importance'].keys()),
                'Importance': list(self.metrics[ticker]['feature_importance'].values())
            }).sort_values('Importance', ascending=False)
            
            # 4. Confusion matrix
            conf_matrix = pd.DataFrame(
                self.metrics[ticker]['confusion_matrix'],
                columns=['Predicted Sell', 'Predicted Hold', 'Predicted Buy'],
                index=['Actual Sell', 'Actual Hold', 'Actual Buy']
            )
            
            # Save all metrics
            base_path = os.path.join(metrics_dir, f'{ticker.replace("/", "_")}')            
            overall_metrics.to_csv(f'{base_path}_overall_metrics.csv', index=False)
            class_metrics_df.to_csv(f'{base_path}_class_metrics.csv', index=False)
            feature_imp.to_csv(f'{base_path}_feature_importance.csv', index=False)
            conf_matrix.to_csv(f'{base_path}_confusion_matrix.csv')
            
            # Save summary to text file
            with open(f'{base_path}_summary.txt', 'w') as f:
                f.write(f"ML Strategy Performance Summary for {ticker}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Overall Metrics:\n")
                f.write("-" * 20 + "\n")
                for _, row in overall_metrics.iterrows():
                    f.write(f"{row['Metric']}: {row['Value']:.4f}\n")
                
                f.write("\nClass-Specific Metrics:\n")
                f.write("-" * 20 + "\n")
                for _, row in class_metrics_df.iterrows():
                    f.write(f"\n{row['Class']}:\n")
                    f.write(f"  Precision: {row['Precision']:.4f}\n")
                    f.write(f"  Recall: {row['Recall']:.4f}\n")
                    f.write(f"  F1-score: {row['F1-score']:.4f}\n")
                    f.write(f"  Support: {row['Support']}\n")
                
                f.write("\nTop 10 Most Important Features:\n")
                f.write("-" * 20 + "\n")
                for _, row in feature_imp.head(10).iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
            
            logging.info(f"Saved detailed metrics for {ticker} to {metrics_dir}")
            
        except Exception as e:
            logging.error(f"Error saving metrics for {ticker}: {str(e)}")
            raise
