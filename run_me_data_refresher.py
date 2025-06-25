#%pip install backtesting
#%pip install nbformat
from assets.DataProvider import DataProvider
from assets.HW2Strategy import HW2Strategy_SMA_RSI, HW2Strategy_SMA
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataResolution, DataPeriod
from backtesting import Strategy
import nbformat
import datetime as dt


resolutions = [DataResolution.HOUR_01, DataResolution.DAY_01, DataResolution.MINUTE_01]

for resolution in resolutions:
    print(f'Processing {resolution} data...')
    # Initialize data provider with BTC/USDT data
    data_provider = DataProvider(
        tickers=['BTC/USDT','ETH/USDT'],
        resolution=resolution,
        period=DataPeriod.YEAR_05)
    
    # Load the data
    data_provider.data_load()
    data_provider.data_refresh()
    data_provider.data_save()

print("Data refresh completed.")
