#%pip install backtesting
#%pip install nbformat
from assets.DataProvider import DataProvider
from assets.HW2Strategy import HW2Strategy_SMA_RSI, HW2Strategy_SMA
from assets.HW2Backtest import HW2Backtest
from assets.enums import DataResolution, DataPeriod
from backtesting import Strategy
import datetime as dt


# Initialize data provider with BTC/USDT data
tickers = ['BTC/USDT']
data_provider = DataProvider(
    tickers=tickers,
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_05
)

# Загружаем локальные данные
data_provider.data_load()
# Построим дашборд для загруженных данных
data_provider.dashboard_data_draw()

# Обновим данные для тикеров
for ticker in tickers:
    data_provider.data_refresh()
data_provider.dashboard_data_draw()

# Проверим фичи
data_provider.dashboard_features_draw()
