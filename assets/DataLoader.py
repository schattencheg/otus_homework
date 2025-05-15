import os
import pandas as pd
import datetime as dt
import ccxt
from typing import Dict, List
from assets.enums import DataPeriod, DataResolution

class DataLoaderBase:
    def __init__(self,  tickers: List[str], 
                        resolution: DataResolution = None, period: DataPeriod = None, 
                        time_start: dt.date = None, time_end: dt.date = None):
        self.tickers: List[str] = tickers
        self.resolution: DataResolution = resolution
        self.period: DataPeriod = period
        self.ts: dt.date = time_start
        self.te: dt.date = time_end
        #
        self.data: Dict[str, pd.DataFrame] = {}

    def data_request(self):
        pass

    def data_request_by_ticker(self, ticker: str, ts: dt.date = None) -> pd.DataFrame:
        pass


class DataLoaderCCXT(DataLoaderBase):
    def __init__(self,  tickers: List[str], 
                        resolution: DataResolution = None, period: DataPeriod = None, 
                        time_start: dt.date = None, time_end: dt.date = None,
                        exchange: str = 'binance'):
        super().__init__(tickers, resolution, period, time_start, time_end)
        self.exchange = getattr(ccxt, exchange)()
        self.timeframe = self._get_timeframe()
        #self.available_symbols = self._get_available_symbols()
        #self._validate_tickers()

    def data_request(self):
        for ticker in self.tickers:
            df = self.data_request_by_ticker(ticker)
            if df is not None:
                df.columns = [x.lower() for x in df.columns]
                df.index.name = df.index.name.lower()
                self.data[ticker] = df

    def data_request_by_ticker(self, ticker: str, ts: dt.date = None, te: dt.date = None) -> pd.DataFrame:
        if ticker in self.data:
            return self.data[ticker]

        try:
            # Convert dates to timestamps
            since = int(ts.timestamp() * 1000) if ts else None
            if since is None:
                since = dt.datetime.now() - dt.timedelta(days=self._get_date_range())
                since = int((dt.datetime.now() - dt.timedelta(days=self._get_date_range())).timestamp())

            until = int(te.timestamp() * 1000) if te else None
            if until is None:
                until = dt.datetime.now()
                until = int(dt.datetime.now().timestamp())

            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=ticker,
                timeframe=self.timeframe,
                since=since,
                limit=1000  # Adjust based on exchange limits
            )

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def _get_timeframe(self) -> str:
        # Convert DataResolution to CCXT timeframe format
        resolution_map = {
            DataResolution.MINUTE_01: '1m',
            DataResolution.MINUTE_02: '2m',
            DataResolution.MINUTE_05: '5m',
            DataResolution.MINUTE_15: '15m',
            DataResolution.MINUTE_30: '30m',
            DataResolution.MINUTE_60: '60m',
            DataResolution.MINUTE_90: '90m',
            DataResolution.DAY_01: '1d',
            DataResolution.DAY_05: '5d',
            DataResolution.MONTH_01: '1mo',
            DataResolution.MONTH_03: '3mo'
        }
        return resolution_map.get(self.resolution, '1d')

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

    def _get_available_symbols(self) -> List[str]:
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            print(f"Error loading markets: {str(e)}")
            return []

    def _validate_tickers(self):
        invalid_tickers = [ticker for ticker in self.tickers if ticker not in self.available_symbols]
        if invalid_tickers:
            print(f"Warning: The following tickers are not available on {self.exchange.name}: {invalid_tickers}")
            self.tickers = [ticker for ticker in self.tickers if ticker not in invalid_tickers]

