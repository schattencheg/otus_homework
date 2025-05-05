from enum import Enum


# period: “1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”
# interval: “1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”

class DataResolution(Enum):
    MINUTE_01 = '1m'
    MINUTE_02 = '2m'
    MINUTE_05 = '5m'
    MINUTE_15 = '15m'
    MINUTE_30 = '30m'
    MINUTE_60 = '60m'
    MINUTE_90 = '90m'
    HOUR = '1h'
    DAY_01 = '1d'
    DAY_05 = '5d'
    WEEK = '1wk'
    MONTH_01 = '1mo'
    MONTH_03 = '3mo'


class DataPeriod(Enum):
    DAY_01 = '1d'
    DAY_05 = '5d'
    MONTH_01 = '1mo'
    MONTH_03 = '3mo'
    MONTH_06 = '6mo'
    YEAR_01 = '1y'
    YEAR_02 = '2y'
    YEAR_05 = '5y'
    YEAR_10 = '10y'
    YEAR_YTD = 'ytd'
    YEAR_MAX = 'max'
