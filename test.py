
import finplot as fplt
import yfinance as yf

df = yf.download('GOOG', interval='90m')

dfd = df.Open.resample('D').first().to_frame()
dfd['Close'] = df.Close.resample('D').last()
dfd['High'] = df.High.resample('D').max()
dfd['Low'] = df.Low.resample('D').min()

daily_plot = fplt.candlestick_ochl(dfd.dropna(), candle_width=5)
daily_plot.colors.update(dict(bull_body='#bfb', bull_shadow='#ada', bear_body='#fbc', bear_shadow='#dab'))
daily_plot.x_offset = 3.1 # resample() gets us start of day

fplt.candlestick_ochl(df[['Open','Close','High','Low']])

fplt.show()