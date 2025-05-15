import datetime as dt
import os
from typing import Dict, List
import pandas as pd
import numpy as np
from pathvalidate import sanitize_filepath
from assets.DataLoader import DataLoaderBase, DataLoaderCCXT
from assets.enums import DataPeriod, DataResolution


class DataProvider:
    def __init__(self,  tickers: List[str] = ['BTC/USDT'], 
                        resolution: DataResolution = DataResolution.DAY_01, 
                        period: DataPeriod = DataPeriod.YEAR_01,
                        ts: dt.date = None,
                        te: dt.date = None):
        self.tickers: List[str] = tickers
        self.tickers_path: Dict[str, str] = {ticker: ticker.replace('/', '_') for ticker in tickers}
        self.resolution: DataResolution = resolution
        self.period: DataPeriod = period
        self.ts: dt.date = ts
        self.te: dt.date = te
        #
        self.data: Dict[str, pd.DataFrame] = {}
        self.data_loader: DataLoaderBase = DataLoaderCCXT(tickers, resolution, period, ts, te)
        # Create Data directory if it doesn't exist
        self.dir_data: str = self.sanitize_path(os.path.join('data', resolution.name))
    
    def data_request(self):
        for ticker in self.tickers:
            self.data[ticker] = self.data_request_by_ticker(ticker)

    def data_request_by_ticker(self, ticker, ts: dt.date = None):
        df: pd.DataFrame = None
        if ticker in self.tickers:
            df = self.data_loader.data_request_by_ticker(ticker, ts)
            if os.path.exists(self.sanitize_path(os.path.join(self.dir_data, self.tickers_path[ticker] + '.csv'))):
                if ticker in self.data and self.data[ticker] is not None:
                    df = pd.concat([df, self.data[ticker]]).drop_duplicates().sort_index()
        return df
    
    def data_refresh(self):
        for ticker in self.tickers:
            ts = self.data[ticker].index[-1]
            self.data_request_by_ticker(ticker, ts)

    def data_load(self):
        for ticker in self.tickers:
            df = self.data_load_by_ticker(ticker)
            if df is not None:
                self.data[ticker] = df
        return bool(self.data)

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

    def drop_nones(self, ticker: str, df:  pd.DataFrame):
        # Удаляем пропуски (None)
        initial_length = len(df)
        df = df.dropna()
        if len(df) != initial_length:
            print(f' Удалено {initial_length - len(df)} пустых значений для {ticker}')
        return df.reset_index(drop=True)

    def drop_high_dispersion(self, ticker: str, df: pd.DataFrame, threshold: float = 0.1) -> None:
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

    def drop_price_anomalies(self, ticker: str, df: pd.DataFrame, 
                                z_threshold: float = 3.0, 
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
        
    def data_clear(self):
        for ticker in self.tickers:
            if ticker in self.data:
                df = self.data[ticker]
                df = self.drop_nones(ticker, df)
                df = self.drop_high_dispersion(ticker, df)
                df = self.drop_price_anomalies(ticker, df)
                return df

    def sanitize_path(self, path):
        return sanitize_filepath(path).lower()
