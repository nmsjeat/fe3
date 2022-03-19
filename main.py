import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read files
file1 = '../quote_20220318.csv'
file2 = '../trade_20220318.csv'

quote = pd.read_csv(file1, sep=',', index_col=0)
trade = pd.read_csv(file2, sep=',', index_col=0)

# Set datetime index
quote.index = pd.to_datetime(quote.index, format="%Y-%m-%dD%H:%M:%S.%f")
trade.index = pd.to_datetime(trade.index, format="%Y-%m-%dD%H:%M:%S.%f")


# Separate into three dataframes based on crypto
xbt_quote = quote[quote['symbol'] == 'XBTUSD']
eth_quote = quote[quote['symbol'] == 'ETHUSD']
bch_quote = quote[quote['symbol'] == 'BCHUSD']

xbt_trade = trade[trade['symbol'] == 'XBTUSD']
eth_trade = trade[trade['symbol'] == 'ETHUSD']
bch_trade = trade[trade['symbol'] == 'BCHUSD']

# Exponential weighted moving average
assets = ['xbt', 'eth', 'bch']

for a in assets:
    a+'_quote' = eval(a+'_quote').assign(BuyPressure=lambda df: (df['bidSize'] / (df['bidSize'] + df['askSize'])).ewm(alpha=0.95).mean())



bch_quote = bch_quote.assign(BuyPressure=lambda df: df['bidSize'] / (df['bidSize'] + df['askSize'])).ewm(alpha=0.95).mean()
bch_quote = bch_quote.assign(BuyPressure=lambda df: df['bidSize'] / (df['bidSize'] + df['askSize'])).ewm(alpha=0.95).mean()


