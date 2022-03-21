import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Resamples data into ival intervals
def resample_df(df_quote, ival):
    df_1 = df_quote.resample(ival).last().rename(columns = {i:('last_'+i if i != 'symbol' else i) for i in df_quote.columns})
    df_2 = df_quote.resample(ival).mean().rename(columns = {i:'mean_'+i for i in df_quote.columns})
    df_3 = df_quote.resample(ival).std(ddof=0).rename(columns = {i:'std_'+i for i in df_quote.columns})
    df_quote = df_1.merge(df_2, left_index=True, right_index=True).merge(df_3, left_index=True, right_index=True)
    return df_quote


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

# Resample data into (1s) intervals
ival = '1s'

xbt_quote = resample_df(xbt_quote, ival)
eth_quote = resample_df(eth_quote, ival)
bch_quote = resample_df(bch_quote, ival)



xbt_1 = xbt_quote.resample(ival).last().rename(columns = {i:('last_'+i if i != 'symbol' else i) for i in xbt_quote.columns})
xbt_2 = xbt_quote.resample(ival).mean().rename(columns = {i:'mean_'+i for i in xbt_quote.columns})
xbt_3 = xbt_quote.resample(ival).std().rename(columns = {i:'std_'+i for i in xbt_quote.columns})
xbt_quote = xbt_1.merge(xbt_2, left_index=True, right_index=True).merge(xbt_3, left_index=True, right_index=True)









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








# xbt_quote.resample(ival).agg({'bidSize': 'sum',
#                               'bidPrice': 'last',
#                               'askSize': lambda x: x.tail(1) - 1
#                               }).rename(columns={'bidSize': 'last_bidSize'})


# y = bch_quote.resample('1s').first().dropna()['bidPrice']
# y = (y.shift(1) - y)[1:]
# x = bch_quote.resample('1s').first().dropna()['MicroPriceAdjustment'][1:]
# x[x.between(-0.05,0.05)] = 0
# x = sm.add_constant(x, has_constant='add')
# model = sm.OLS(y,x).fit()
# model.summary()




