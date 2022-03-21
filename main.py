import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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
xbt_quote = xbt_quote.assign(BuyPressure=lambda df: (df['bidSize'] / (df['bidSize'] + df['askSize'])).ewm(alpha=0.95).mean())
eth_quote = eth_quote.assign(BuyPressure=lambda df: (df['bidSize'] / (df['bidSize'] + df['askSize'])).ewm(alpha=0.95).mean())
bch_quote = bch_quote.assign(BuyPressure=lambda df: (df['bidSize'] / (df['bidSize'] + df['askSize'])).ewm(alpha=0.95).mean())

# Weighted mid-price
xbt_quote = xbt_quote.assign(WeightedMid=lambda df: df['BuyPressure'] * df['askPrice'] + (1 - df['BuyPressure']) * df['bidPrice'])
eth_quote = eth_quote.assign(WeightedMid=lambda df: df['BuyPressure'] * df['askPrice'] + (1 - df['BuyPressure']) * df['bidPrice'])
bch_quote = bch_quote.assign(WeightedMid=lambda df: df['BuyPressure'] * df['askPrice'] + (1 - df['BuyPressure']) * df['bidPrice'])

# Micro price adjustment
xbt_quote = xbt_quote.assign(MicroPriceAdjustment=lambda df: df['WeightedMid'] - 0.5 * (df['askPrice'] + df['bidPrice']))
eth_quote = eth_quote.assign(MicroPriceAdjustment=lambda df: df['WeightedMid'] - 0.5 * (df['askPrice'] + df['bidPrice']))
bch_quote = bch_quote.assign(MicroPriceAdjustment=lambda df: df['WeightedMid'] - 0.5 * (df['askPrice'] + df['bidPrice']))








y = bch_quote.resample('1s').first().dropna()['bidPrice']
y = (y.shift(1) - y)[1:]
x = bch_quote.resample('1s').first().dropna()['MicroPriceAdjustment'][1:]
x[x.between(-0.05,0.05)] = 0
x = sm.add_constant(x, has_constant='add')
model = sm.OLS(y,x).fit()
model.summary()




