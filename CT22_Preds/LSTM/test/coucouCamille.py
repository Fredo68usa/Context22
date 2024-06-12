import ccxt
import pandas as pd

ex = ccxt.binance()

# download data from binance spot market
df = pd.DataFrame(
    ex.fetch_ohlcv(symbol='BTCUSDT', timeframe='1d', limit=1000), 
    columns = ['unix', 'open', 'high', 'low', 'close', 'volume']
)

# convert unix (in milliseconds) to UTC time
df['date'] = pd.to_datetime(df.unix, unit='ms')

