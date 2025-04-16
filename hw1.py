#%pip install yfinance -qq
#%pip install pandas_datareader pandas numpy plotly seaborn matplotlib scipy -qq
from scipy.stats import zscore
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader as web
import plotly.graph_objects as go
import seaborn as sns
import yfinance as yf

class hw1:
    def __init__(self):
        # Create Data directory if it doesn't exist
        self.dir_stocks = os.path.join('data', 'stock')
        if not os.path.exists(self.dir_stocks):
            os.makedirs(self.dir_stocks)
        self.dir_crypto = os.path.join('data', 'crypto')
        if not os.path.exists(self.dir_crypto):
            os.makedirs(self.dir_crypto)

    def download_data(self, tickers: List[str] = None, cryptos: List[str] = None):
        if tickers is None:
            # Download SnP500 sub-tickers data
            tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].str.replace('.','-').to_list()[:10]
        data: Dict[str, pd.DataFrame] = yf.download(tickers=tickers, group_by='Ticker', multi_level_index=False,progress=False)
        data_per_ticker = {}
        for ticker in tickers:
            df = data[ticker]#.dropna()
            df.columns = [x.lower() for x in df.columns]
            df.index.name = df.index.name.lower()
            df.to_csv(os.path.join(dir_stocks, ticker + '.csv'))



        if cryptos is None:
            # Download crypto data
            cryptos = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
        for crypto in cryptos:
            data = yf.download(crypto, multi_level_index=False, progress=False)#.dropna()
            df.columns = [x.lower() for x in df.columns]
        df.index.name = df.index.name.lower()
        data.to_csv(os.path.join(dir_crypto, crypto.split("-")[0] + '.csv'))

