import datetime as dt
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathvalidate import sanitize_filepath
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from assets.DataLoader import DataLoaderBase, DataLoaderCCXT
from assets.enums import DataPeriod, DataResolution

logging.basicConfig(level=logging.INFO)


class DataProvider:
    def __init__(self,  tickers: List[str] = ['BTC/USDT'], 
                        resolution: DataResolution = DataResolution.DAY_01, 
                        period: DataPeriod = DataPeriod.YEAR_01,
                        ts: dt.date = None,
                        te: dt.date = None,
                        outlier_std_threshold: float = 5.0,
                        gap_fill_limit: int = 20,
                        skip_dashboard: bool = False):
        self.tickers: List[str] = tickers
        self.tickers_path: Dict[str, str] = {ticker: ticker.replace('/', '_') for ticker in tickers}
        self.resolution: DataResolution = resolution
        self.period: DataPeriod = period
        self.ts: dt.date = ts
        self.te: dt.date = te
        #
        self.data: Dict[str, pd.DataFrame] = {}
        self.data_processed: Dict[str, pd.DataFrame] = {}
        self.data_loader: DataLoaderBase = DataLoaderCCXT(tickers, resolution, period, ts, te)
        self.outlier_std_threshold = outlier_std_threshold
        self.gap_fill_limit = gap_fill_limit
        self.dashboard_data: go.Figure = None
        self.dashboard_features: go.Figure = None
        # Create necessary directories
        self.dir_data: str = self.sanitize_path(os.path.join('data', resolution.name))
        self.dir_metrics: str = self.sanitize_path(os.path.join('data', 'metrics'))
        self.dir_dashboard: str = self.sanitize_path('dashboard')
        for directory in [self.dir_data, self.dir_metrics, self.dir_dashboard]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        # Initialize metrics tracking
        self.metrics_file = os.path.join(self.dir_metrics, 'data_quality_metrics.csv')
        if os.path.exists(self.metrics_file):
            self.metrics_history = pd.read_csv(self.metrics_file)
        else:
            self.metrics_history = pd.DataFrame(columns=[
                'timestamp', 'ticker', 'total_rows', 'missing_values',
                'outliers', 'gaps_filled', 'high_dispersion'
            ])

#region Feature Engineering        
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        if df is None or df.empty:
            return df
        # Price features
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        # Volatility features
        df['ATR'] = self.calculate_atr(df)
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Std'] = df['Volume'].rolling(window=20).std()
            df['Volume_ZScore'] = (df['Volume'] - df['Volume_MA']) / df['Volume_Std']
        # Technical indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
#endregion

#region Dashboard
    def record_metrics(self, ticker: str, df: pd.DataFrame, initial_rows: int, initial_missing: int):
        # Record data quality metrics
        current_rows = len(df) if df is not None else 0
        current_missing = df.isnull().sum().sum() if df is not None else 0
        
        new_metrics = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'ticker': [ticker],
            'total_rows': [current_rows],
            'missing_values': [current_missing],
            'outliers': [initial_rows - current_rows if initial_rows > current_rows else 0],
            'gaps_filled': [initial_missing - current_missing if initial_missing > current_missing else 0],
            'high_dispersion': [0]  # This will be updated by clean_data
        })
        
        self.metrics_history = pd.concat([self.metrics_history, new_metrics], ignore_index=True)
        self.metrics_history.to_csv(self.metrics_file, index=False)
        
        return new_metrics

    def update_dashboard(self, ticker: str, df: pd.DataFrame) -> None:
        """Update dashboard for a ticker"""
        if df is None or df.empty:
            return
        # Create subplots
        self.dashboard_data = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price and Volume',
                'Technical Indicators',
                'Data Quality Metrics',
                'Returns Distribution',
                'Feature Correlations',
                'Missing Data Points'
            ),
            vertical_spacing=0.12)
        # 1. Price and Volume
        self.dashboard_data.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'),
            row=1, col=1)
        if 'Volume' in df.columns:
            self.dashboard_data.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                row=2, col=2
            )
        # 2. Technical Indicators
        self.dashboard_data.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=1, col=2)
        self.dashboard_data.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=1, col=2)
        # 3. Data Quality Metrics
        if not self.metrics_history.empty:
            ticker_metrics = self.metrics_history[self.metrics_history['ticker'] == ticker]
            if not ticker_metrics.empty:
                metrics = ticker_metrics.iloc[-1]
                for metric in ['total_rows', 'missing_values', 'outliers', 'gaps_filled', 'high_dispersion']:
                    if metric in metrics:
                        self.dashboard_data.add_trace(
                            go.Scatter(
                                x=[metrics['timestamp']],
                                y=metrics[metric],
                                name=metric,
                                mode='lines+markers'
                            ),
                            row=2, col=1
                        )
        # 4. Returns Distribution
        self.dashboard_data.add_trace(
            go.Histogram(x=df['Returns'], name='Returns Distribution', nbinsx=50),
            row=2, col=2)
        # 5. Feature Correlations
        corr = df[['Close', 'Volume', 'RSI', 'MACD', 'Volatility']].corr()
        self.dashboard_data.add_trace(
            go.Heatmap(z=corr.values, x=corr.index, y=corr.columns, name='Correlations'),
            row=3, col=1)
        # 6. Missing Data Points
        missing = df.isnull().sum(axis=1)
        self.dashboard_data.add_trace(
            go.Scatter(x=df.index, y=missing, name='Missing Points'),
            row=3, col=2)
        # Update layout
        self.dashboard_data.update_layout(
            height=1200,
            width=1600,
            title_text=f'Data Analysis Dashboard - {ticker}',
            showlegend=True)
        # Save dashboard
        path = os.path.join(self.dir_dashboard, f'{self.tickers_path[ticker]}_dashboard.html')
        self.dashboard_data.write_html(path)
        self.dashboard_data_draw()

    def create_data_dashboard(self, ticker: str) -> None:
        """Create dashboard showing data accumulation and quality metrics"""
        if ticker not in self.data_processed or self.data_processed[ticker] is None:
            print(f"No processed data available for {ticker}")
            return
        df = self.data_processed[ticker]
        metrics = self.metrics_history[self.metrics_history['ticker'] == ticker]
        # Create subplots
        self.dashboard_data = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price History',
                'Data Quality Metrics',
                'Technical Indicators',
                'Volume Profile',
                'Volatility',
                'Missing Data Points'))
        # 1. Price History
        self.dashboard_data.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'),
            row=1, col=1)
        # 2. Data Quality Metrics
        for metric in ['missing_values', 'outliers', 'high_dispersion']:
            self.dashboard_data.add_trace(
                go.Scatter(
                    x=metrics['timestamp'],
                    y=metrics[metric],
                    name=metric),
                row=1, col=2)
        # 3. Technical Indicators
        self.dashboard_data.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'),
            row=2, col=1)
        self.dashboard_data.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
            row=2, col=1)
        # 4. Volume Profile
        if 'Volume' in df.columns:
            self.dashboard_data.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                row=2, col=2)
        # 5. Volatility
        self.dashboard_data.add_trace(
            go.Scatter(x=df.index, y=df['ATR'], name='ATR'),
            row=3, col=1)
        # 6. Missing Data Points
        missing_data = df.isnull().sum(axis=1)
        self.dashboard_data.add_trace(
            go.Scatter(x=df.index, y=missing_data, name='Missing Points'),
            row=3, col=2)
        # Update layout
        self.dashboard_data.update_layout(
            height=1200,
            width=1600,
            title_text=f"Data Analysis Dashboard - {ticker}",
            showlegend=True)
        # Save the dashboard
        self.dashboard_data.write_html(os.path.join(self.dir_metrics, f'{ticker.replace("/", "_")}_dashboard.html'))

    def create_features_dashboard(self, ticker: str) -> None:
        """Create dashboard showing technical analysis features"""
        if ticker not in self.data_processed or self.data_processed[ticker] is None:
            print(f"No processed data available for {ticker}")
            return

        df = self.data_processed[ticker]
        
        # Create subplots
        self.dashboard_features = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Price and Moving Averages',
                'Returns Distribution',
                'RSI and MACD',
                'Volume Analysis',
                'Volatility Metrics',
                'Moving Average Comparison',
                'Log Returns',
                'Volume Z-Score'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1)

        # 1. Price and Moving Averages
        self.dashboard_features.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'),
            row=1, col=1)
        
        # Add SMA lines
        for period in [20, 50, 200]:
            self.dashboard_features.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'SMA_{period}'],
                    name=f'SMA {period}'),
                row=1, col=1)

        # 2. Returns Distribution
        self.dashboard_features.add_trace(
            go.Histogram(
                x=df['Returns'],
                name='Returns Distribution',
                nbinsx=50),
            row=1, col=2)

        # 3. RSI and MACD
        self.dashboard_features.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
            row=2, col=1)
        self.dashboard_features.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=2, col=1)
        self.dashboard_features.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal'),
            row=2, col=1)

        # 4. Volume Analysis
        if 'Volume' in df.columns:
            self.dashboard_features.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                row=2, col=2)
            self.dashboard_features.add_trace(
                go.Scatter(x=df.index, y=df['Volume_MA'], name='Volume MA'),
                row=2, col=2)

        # 5. Volatility Metrics
        self.dashboard_features.add_trace(
            go.Scatter(x=df.index, y=df['ATR'], name='ATR'),
            row=3, col=1)
        self.dashboard_features.add_trace(
            go.Scatter(x=df.index, y=df['Volatility'], name='Volatility'),
            row=3, col=1)

        # 6. Moving Average Comparison
        for period in [5, 10, 20]:
            self.dashboard_features.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'EMA_{period}'] - df[f'SMA_{period}'],
                    name=f'EMA-SMA {period}'),
                row=3, col=2)

        # 7. Log Returns
        self.dashboard_features.add_trace(
            go.Scatter(x=df.index, y=df['Log_Returns'], name='Log Returns'),
            row=4, col=1)

        # 8. Volume Z-Score
        if 'Volume_ZScore' in df.columns:
            self.dashboard_features.add_trace(
                go.Scatter(x=df.index, y=df['Volume_ZScore'], name='Volume Z-Score'),
                row=4, col=2)

        # Update layout
        self.dashboard_features.update_layout(
            height=1600,
            width=1600,
            title_text=f"Technical Analysis Features Dashboard - {ticker}",
            showlegend=True)

        # Save the dashboard
        self.dashboard_features.write_html(os.path.join(self.dir_dashboard, f'{self.tickers_path[ticker]}_features_dashboard.html'))

    def dashboard_data_draw(self):
        self.dashboard_data.update_layout(height=800, width=1200, title_text='')
        self.dashboard_data.show()

    def dashboard_features_draw(self):
        self.dashboard_features.update_layout(height=800, width=1200, title_text='')
        self.dashboard_features.show()
#endregion

#region Utility
    def sanitize_path(self, path):
        return sanitize_filepath(path).lower()

    def get_frequency(self) -> str:
        """Get pandas frequency string based on resolution"""
        resolution_map = {
            DataResolution.MINUTE_01: '1min',
            DataResolution.MINUTE_05: '5min',
            DataResolution.MINUTE_15: '15min',
            DataResolution.HOUR_01: '1H',
            DataResolution.DAY_01: '1D',
            DataResolution.WEEK_01: '1W'
        }
        return resolution_map.get(self.resolution, '1D')
#endregion

#region Data Processing
    def data_load(self):
        for ticker in self.tickers:
            df = self.data_load_by_ticker(ticker)
            if df is not None:
                self.data[ticker] = df
                self.data_processed[ticker] = self.process_new_data(ticker, df)
                # Update dashboard
                self.create_data_dashboard(ticker)
                self.create_features_dashboard(ticker)
        return bool(self.data_processed)

    def data_load_by_ticker(self, ticker):
        df: pd.DataFrame = None
        if ticker in self.tickers:
            if os.path.exists(self.sanitize_path(os.path.join(self.dir_data, self.tickers_path[ticker] + '.csv'))):
                df = pd.read_csv(self.sanitize_path(os.path.join(self.dir_data, self.tickers_path[ticker] + '.csv')), index_col=0)
                # Convert index to datetime
                df.index = pd.to_datetime(df.index)
        return df

    def data_save(self):
        for ticker in self.tickers:
            self.data_save_by_ticker(ticker)

    def data_save_by_ticker(self, ticker):
        if not os.path.exists(self.dir_data):
            os.makedirs(self.dir_data)
        self.data[ticker].to_csv(self.sanitize_path(os.path.join(self.dir_data, self.tickers_path[ticker] + '.csv')), index=True)

    def data_save_by_ticker_and_df(self, ticker, df: pd.DataFrame):
        if not os.path.exists(self.dir_data):
            os.makedirs(self.dir_data)
        df.to_csv(self.sanitize_path(os.path.join(self.dir_data, self.tickers_path[ticker] + '.csv')), index=True)

    def process_data(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """Process data through cleaning and feature engineering pipeline"""
        if df is None or df.empty:
            return df
        df.columns = [x.capitalize() for x in df.columns]
        # Store initial state for metrics
        initial_rows = len(df)
        initial_missing = df.isnull().sum().sum()
        # 1. Clean data
        df = self.clean_data_for_ticker(ticker, df)
        # 2. Add features
        df = self.add_features(df)
        # Record metrics
        self.record_metrics(ticker, df, initial_rows, initial_missing)
        return df
        
    def clean_data(self):
        for ticker in self.tickers:
            df = self.data[ticker]
            df = self.clean_data_for_ticker(ticker, df)
            self.data[ticker] = df

    def clean_data_for_ticker(self, ticker: str = None, df: pd.DataFrame = None):
        if df is None:
            return None
            
        initial_rows = len(df)
        initial_missing = df.isnull().sum().sum()
        
        def drop_nones(ticker: str, df: pd.DataFrame):
            if df is None or df.empty:
                return df
            df = df.dropna()
            rows_dropped = initial_rows - len(df)
            if rows_dropped > 0:
                logging.info(f'Dropped {rows_dropped} rows with None values for {ticker}')
            return df

        def drop_high_dispersion(ticker: str, df: pd.DataFrame, threshold: float = 0.5) -> None:
            # Удаляем данные с высокой дисперсией данных, (High - Low) / ((High + Low) / 2) > threshold
            dispersion = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)
            # Создаём фильтр
            valid_mask = dispersion <= threshold
            # Применяем фильтр
            df = df[valid_mask]
            removed_count = (~valid_mask).sum()
            if removed_count > 0:
                print(f" Удалено {removed_count} записей с высокой дисперсией для {ticker}")
            return df

        def drop_price_anomalies(ticker: str, df: pd.DataFrame, 
                                    z_threshold: float = 5.0, 
                                    window: int = 20) -> None:
            # Удаляем данные, применяя z-score для МА(window) (если данные более чем на 3 стандартных отклонения - в мусор)
            price_columns = ['Open', 'High', 'Low', 'Close']
            # Создаём фильтр
            valid_mask = pd.Series(True, index=df.index)
            for col in price_columns:
                # Рассчитываем МА(window)
                rolling_mean = df[col].rolling(window=window, center=True).mean()
                rolling_std = df[col].rolling(window=window, center=True).std()
                # Рассчет отклонения
                z_scores = np.abs((df[col] - rolling_mean) / rolling_std)
                # Убираем из фильтра неподходящие значения
                valid_mask &= (z_scores <= z_threshold)
            # Применяем фильтр
            df = df[valid_mask]
            removed_count = (~valid_mask).sum()
            if removed_count > 0:
                print(f" Удалено {removed_count} аномальных цен {ticker}")
            return df

        def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
            """Fill gaps in time series data"""
            if df is None or df.empty:
                return df
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            # Create complete date range
            freq = self.get_frequency()
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
            # Reindex and forward fill small gaps
            df = df.reindex(date_range)
            df = df.ffill(limit=self.gap_fill_limit)
            # For remaining gaps, use linear interpolation
            df = df.interpolate(method='linear', limit=self.gap_fill_limit)
            return df

        """Clean data from outliers and invalid values"""
        if df is None or df.empty:
            return df
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        # Handle missing values
        df = drop_nones(ticker, df)
        # Remove high dispersion data
        df = drop_high_dispersion(ticker, df)
        # Remove price anomalies
        df = drop_price_anomalies(ticker, df)
        # Fill gaps
        df = fill_gaps(df)
        return df

    def data_request(self, ts: dt.date = None):
        for ticker in self.tickers:
            try:
                new_data = self.data_request_by_ticker(ticker, ts)
                if new_data is not None and not new_data.empty:
                    self.data[ticker] = new_data
                    self.data_processed[ticker] = self.process_new_data(ticker, new_data)
                    # Record initial metrics
                    self.record_metrics(ticker, new_data, len(new_data), new_data.isnull().sum().sum())
            except Exception as e:
                logging.error(f"Error processing {ticker}: {str(e)}")
                continue

    def data_request_by_ticker(self, ticker, ts: dt.date = None) -> Optional[pd.DataFrame]:
        """Request new data for a ticker and process it through the pipeline"""
        if ticker not in self.tickers:
            return None

        if ticker in self.data:
            return self.data[ticker]

        # Get new data
        new_data = self.data_loader.data_request_by_ticker(ticker, ts)
        if new_data is None or new_data.empty:
            return None
        return new_data

    def data_refresh(self):
        for ticker in self.tickers:
            ts = self.data[ticker].index[-1]
            new_data = self.data_request_by_ticker(ticker, ts)
            if new_data is not None and not new_data.empty:
                existing_data = self.data[ticker] if ticker in self.data else None
                self.data[ticker] = self.data_append(existing_data, new_data)
                self.data_processed[ticker] = self.process_new_data(ticker, new_data)
                # Update dashboards
                self.create_data_dashboard(ticker)
                self.create_features_dashboard(ticker)

    def process_new_data(self, ticker, new_data: pd.DataFrame):
        # Load existing data if any
        existing_data = None
        if ticker in self.data:
            existing_data = self.data[ticker]
            if existing_data is not None:
                existing_data.columns = [x.capitalize() for x in existing_data.columns]
        # Process new data
        processed_data = self.process_data(ticker, new_data)
        # Merge with existing data if any
        processed_data = self.data_append(existing_data, processed_data)
        return processed_data

    def data_append(self, existing_data, new_data):
        if existing_data is not None and not existing_data.empty:
            new_data = pd.concat([existing_data, new_data])
            new_data = new_data[~new_data.index.duplicated(keep='last')]
            new_data = new_data.sort_index()
        return new_data

    def _get_date_range(self) -> str:
        days=0
        if self.period == DataPeriod.DAY_01:
            days = 1
        elif self.period == DataPeriod.DAY_05:
            days = 5
        elif self.period == DataPeriod.MONTH_01:
            days = 4 * 7
        elif self.period == DataPeriod.MONTH_03:
            days = 3 * 4 * 7
        elif self.period == DataPeriod.MONTH_06:
            days = 6 * 4 * 7
        elif self.period == DataPeriod.YEAR_01:
            days = 12 * 4 * 7
        elif self.period == DataPeriod.YEAR_02:
            days = 24 * 4 * 7
        elif self.period == DataPeriod.YEAR_05:
            days = 60 * 4 * 7
        elif self.period == DataPeriod.YEAR_10:
            days = 120 * 4 * 7
        return days
#endregion
