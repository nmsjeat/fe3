import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Resamples data into ival intervals
def resample_df(df_quote, df_trade, ival='1s'):
    """

    Parameters
    ----------
    df_quote : pandas.DataFrame
        Data frame with quotes
    df_trade : pandas.DataFrame
        Data frame with trades
    ival : str
        Resampling interval. Defaults to 1 second but can be altered

    Returns
    -------
    df : pandas.DataFrame
        Merged dataframe with all features

    """
    
    # Record last value for each feature quote data and name as last_feature 
    df_1 = df_quote.resample(ival).last().rename(columns = {i:('last_'+i if i != 'symbol' else i) for i in df_quote.columns})
    # Record mean value for each numeric feature in quote data and name as mean_feature
    df_2 = df_quote.resample(ival).mean().rename(columns = {i:'mean_'+i for i in df_quote.columns})
    # Record standard deviation for each numeric feature in quote data and name as std_feature
    df_3 = df_quote.resample(ival).std().rename(columns = {i:'std_'+i for i in df_quote.columns})
    # Merge newly created last-, mean-, and std-features as final quote data frame
    df_quote = df_1.merge(df_2, left_index=True, right_index=True, how='outer').merge(df_3, left_index=True, right_index=True, how='outer').dropna()

    # Record last value for side, size and price features in trade data and name as last_feature
    df_4 = df_trade.resample(ival).last()[['side', 'size', 'price']].rename(columns = {i:'last_'+i for i in df_trade.columns})
    # Record mean value for size and price features in trade data and name as mean_feature
    df_5 = df_trade.resample(ival).mean()[['size', 'price']].rename(columns = {i:'mean_'+i for i in df_trade.columns})
    # Record standard deviation for size and price features in trade data and name as std_feature
    df_6 = df_trade.resample(ival).std(ddof=0)[['size', 'price']].rename(columns = {i:'std_'+i for i in df_trade.columns})    

    # Count the number of buys and their volume, and name as buys and buyVolume
    df_7 = df_trade[df_trade['side'] == 'Buy'].resample(ival).agg({'side':'count', 'size':'sum'}).rename(columns={'side':'buys', 'size':'buyVolume'})
    # Count the number of sells and their volume, and name as sells and sellVolume
    df_8 = df_trade[df_trade['side'] == 'Sell'].resample(ival).agg({'side':'count', 'size':'sum'}).rename(columns={'side':'sells', 'size':'sellVolume'})
    # Merge buy and sell data to quote data
    df_quote = df_quote.merge(df_7, left_index=True, right_index=True, how='left').merge(df_8, left_index=True, right_index=True, how='left').fillna(0)

    # Merge last-, mean-, and std-features as final trade data frame
    df_trade = df_4.merge(df_5, left_index=True, right_index=True, how='outer').merge(df_6, left_index=True, right_index=True, how='outer')

    # Merge trade data frame with quote data frame as master data frame
    df = df_quote.merge(df_trade, left_index=True, right_index=True, how='left')
    
    # return master data frame
    return df


def micro_price_adjustment(df, alpha=0.95, method='mean'):
    """

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with quote data
    alpha : float
        Parameter for EWMA calculations. The default is 0.95.
    method : string
        Specifies whether 'mean' or 'last' features are used. The default is 'mean'.

    Returns
    -------
    df : pandas.DataFrame
        Data frame that includes the calculated micro price adjustment

    """
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

# Resample data and do feature engineering
xbt = resample_df(xbt_quote, xbt_trade, ival)
eth = resample_df(eth_quote, eth_trade, ival)
bch = resample_df(bch_quote, bch_trade, ival)

# Calculate micro price adjustments
xbt = micro_price_adjustment(xbt, alpha=.95, method='mean')
eth = micro_price_adjustment(eth, alpha=.95, method='mean')
bch = micro_price_adjustment(bch, alpha=.95, method='mean')


# Linear Regression

# Predict change in last best bid price with the micro price adjustment

y = bch['last_bidPrice']
y = (y.shift(1) - y)[1:]
x = bch['MicroPriceAdjustment'][1:]
x[x.between(-0.05,0.05)] = 0
x = sm.add_constant(x, has_constant='add')
model = sm.OLS(y,x).fit()
model.summary()
