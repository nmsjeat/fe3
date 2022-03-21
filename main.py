import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Resamples data into ival intervals
def resample_df(df_quote, df_trade, ival):
    df_1 = df_quote.resample(ival).last().rename(columns = {i:('last_'+i if i != 'symbol' else i) for i in df_quote.columns})
    df_2 = df_quote.resample(ival).mean().rename(columns = {i:'mean_'+i for i in df_quote.columns})
    df_3 = df_quote.resample(ival).std().rename(columns = {i:'std_'+i for i in df_quote.columns})
    df_quote = df_1.merge(df_2, left_index=True, right_index=True, how='outer').merge(df_3, left_index=True, right_index=True, how='outer').dropna()

    df_4 = df_trade.resample(ival).last()[['side', 'size', 'price']].rename(columns = {i:'last_'+i for i in df_trade.columns})
    df_5 = df_trade.resample(ival).mean()[['size', 'price']].rename(columns = {i:'mean_'+i for i in df_trade.columns})
    df_6 = df_trade.resample(ival).std(ddof=0)[['size', 'price']].rename(columns = {i:'std_'+i for i in df_trade.columns})

    df_7 = df_trade[df_trade['side'] == 'Buy'].resample(ival).agg({'side':'count', 'size':'sum'}).rename(columns={'side':'buys', 'size':'buyVolume'})
    df_8 = df_trade[df_trade['side'] == 'Sell'].resample(ival).agg({'side':'count', 'size':'sum'}).rename(columns={'side':'sells', 'size':'sellVolume'})
    df_quote = df_quote.merge(df_7, left_index=True, right_index=True, how='left').merge(df_8, left_index=True, right_index=True, how='left').fillna(0)

    df_trade = df_4.merge(df_5, left_index=True, right_index=True, how='outer').merge(df_6, left_index=True, right_index=True, how='outer')

    df = df_quote.merge(df_trade, left_index=True, right_index=True, how='left')
    return df

def micro_price_adjustment(df, alpha=0.95, method='mean'):
    # Exponential weighted moving average
    df = df.assign(BuyPressure=lambda df: (df[method+'_bidSize'] / (df[method+'_bidSize'] + df[method+'_askSize'])).ewm(alpha=alpha).mean())
    
    # Weighted mid-price
    df = df.assign(WeightedMid=lambda df: df['BuyPressure'] * df[method+'_askPrice'] + (1 - df['BuyPressure']) * df[method+'_bidPrice'])
    
    # Micro price adjustment
    df = df.assign(MicroPriceAdjustment=lambda df: df['WeightedMid'] - 0.5 * (df[method+'_askPrice'] + df[method+'_bidPrice']))
    return df


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

xbt = resample_df(xbt_quote, xbt_trade, ival)
eth = resample_df(eth_quote, eth_trade, ival)
bch = resample_df(bch_quote, bch_trade, ival)

xbt = micro_price_adjustment(xbt)
eth = micro_price_adjustment(eth)
bch = micro_price_adjustment(bch)







# y = bch['last_bidPrice']
# y = (y.shift(1) - y)[1:]
# x = bch['MicroPriceAdjustment'][1:]
# # x[x.between(-0.05,0.05)] = 0
# x = sm.add_constant(x, has_constant='add')
# model = sm.OLS(y,x).fit()
# model.summary()
