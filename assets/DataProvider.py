import datetime as dt
import os
from typing import Dict, List
import pandas as pd
import yfinance as yf
from enums import DataPeriod, DataResolution

class DataProvider:
    def __init__(self,  tickers: List[str] = ['BTC-USD'], 
                        resolution: DataResolution = DataResolution.DAY_01, 
                        period: DataPeriod = DataPeriod.YEAR_MAX,
                        ts: dt.date = None,
                        te: dt.date = None):
        self.tickers: List[str] = tickers
        self.resolution: DataResolution = resolution
        self.period: DataPeriod = period
        self.ts: dt.date = ts
        self.te: dt.date = te
        #
        self.data: Dict[str, pd.DataFrame] = {}
        # Create Data directory if it doesn't exist
        self.dir_data: str = os.path.join('data', resolution.name)

    def data_request(self):
        for ticker in self.tickers:
            df = get_data_by_ticker(ticker)
            df.columns = [x.lower() for x in df.columns]
            df.index.name = df.index.name.lower()
            df.to_csv(os.path.join(self.dir_data, ticker + '.csv'))
            self.data[ticker] = df

    def data_request_by_ticker(self, ticker: str, ts: dt.date = None, te: dt.date = None) -> pd.DataFrame:
        if self.ts is None:
            df = yf.download(ticker, multi_level_index=False, progress=False, 
                                period=self.period.value, interval=self.resolution.value)
        else:
            if te is None:
                te = dt.datetime.now().date()
            df = yf.download(ticker, multi_level_index=False, progress=False, 
                                start=self.ts, end=self.te, interval=self.resolution.value)
        df.columns = [x.lower() for x in df.columns]
        df.index.name = df.index.name.lower()
        return df

    def data_refresh(self):
        for ticker in self.tickers:
            ts = self.data[ticker].index[-1]
            self.data_request_by_ticker(ticker, ts)

    def data_load(self):
        for ticker in self.tickers:
            self.data_load_by_ticker(ticker)

    def data_load_by_ticker(self, ticker):
        if ticker not in self.tickers:
            if os.path.exists(os.path.join(self.dir_data, ticker + '.csv')):
                self.data[ticker] = pd.read_csv(os.path.join(self.dir_data, ticker + '.csv'), index_col=0)
        return self.data[ticker]

    def data_save(self):
        for ticker in self.tickers:
            self.data_save_by_ticker(ticker)

    def data_save_by_ticker(self, ticker):
        if not os.path.exists(self.dir_data):
            os.makedirs(self.dir_data)
        self.data[ticker].to_csv(os.path.join(self.dir_data, ticker + '.csv'), index=True)

    def data_clear(self):
        for ticker in self.tickers:
            if ticker in self.data:
                df = self.data[ticker]
                df = drop_nones(ticker, df)
                df = drop_high_dispersion(ticker, df)
                df = drop_price_anomalies(ticker, df)
                return df

    def drop_nones(self, ticker: str, df:  pd.DataFrame):
        # Удаляем пропуски (None)
        initial_length = len(df)
        df = df.dropna()
        if len(df) != initial_length:
            print(f' Удалено {initial_length - len(df)} пустых значений для {ticker}')
        return df.reset_index(drop=True)

    def drop_high_dispersion(self, ticker: str, df: pd.DataFrame, threshold: float = 0.1) -> None:
        # Удаляем данные с высокой дисперсией данных, (High - Low) / ((High + Low) / 2) > threshold
        dispersion = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
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
        price_columns = ['open', 'high', 'low', 'close']
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

if __name__ == '__main__':
    dp = DataProvider()
    dp.data_request_by_ticker('BTC-USD')
