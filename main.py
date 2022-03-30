import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import minmax_scale
from matplotlib import rc_file_defaults as plotdefaults


def resample_df(df_quote, df_trade, ival='1s'):
    """
    - Resamples data into ival intervals
    - Creates features
    - Merges trade and quotedata

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
    Calculates the micro price adjustment
    
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

def relative_bid_ask_spread(df, method='last'):
    # Relative spread between bid and ask
    df = df.assign(RelativeSpread=lambda df: (df['last_askPrice'] - df['last_bidPrice']) / df['WeightedMid'])
    
    return df

def relative_buys(df):
    # Relative buys to sells, NaN if no trades
    df = df.assign(RelativeBuys=lambda df: 2*(df['buys'] / (df['buys'] + df['sells']))-1)
    df['RelativeBuys'] = df['RelativeBuys'].fillna(0)
    return df

def sales_dummy(df):
    # Sales dummy, 1 if there was sales during last time period, else 0
    df['SalesDummy'] = [0 if x + y == 0 else 1 for (x, y) in df[['buys', 'sells']].values]
    
    return df

def relative_smas(df, method='mean', short=60, long=300):
    df['BidSizeSMA'] = df[method+'_bidSize'].rolling(short, min_periods=1).mean() / df[method+'_bidSize']
    df['AskSizeSMA'] = df[method+'_askSize'].rolling(short, min_periods=1).mean() / df[method+'_askSize']
    
    df['BidPriceSMA_s'] = df[method+'_bidPrice'].rolling(short, min_periods=1).mean() / df[method+'_bidPrice']
    df['AskPriceSMA_s'] = df[method+'_askPrice'].rolling(short, min_periods=1).mean() / df[method+'_askPrice']
    
    df['BidPriceSMA_l'] = df[method+'_bidPrice'].rolling(long, min_periods=1).mean() / df[method+'_bidPrice']
    df['AskPriceSMA_l'] = df[method+'_askPrice'].rolling(long, min_periods=1).mean() / df[method+'_askPrice']
    
    return df

def y_vars(df):
    # Define ticksizes
    ticksize = {'XBTUSD':0.5, 'ETHUSD':0.05, 'BCHUSD':0.05}
    symbol = df.iloc[0].at['symbol']
    
    df['y_changeBidPrice'] = (df['last_bidPrice'] - df['last_bidPrice'].shift(1)).shift(-1) / ticksize[symbol]
    df['y_changeAskPrice'] = (df['last_askPrice'] - df['last_askPrice'].shift(1)).shift(-1) / ticksize[symbol]

    df['y_bidTickDown'] = np.sign(df['last_bidPrice'].diff().shift(-1)).replace({1:0, -1:1})
    df['y_askTickUp'] = np.sign(df['last_askPrice'].diff().shift(-1)).replace({-1:0})
    
    df['y_bidSize'] = df['last_bidSize'].shift(-1)
    df['y_askSize'] = df['last_askSize'].shift(-1)
    
    df['y_sells'] = df['sells'].shift(-1)
    df['y_buys'] = df['buys'].shift(-1)
    
    return df

def clean_columns(df):
    # Define ticksizes
    ticksize = {'XBTUSD':0.5, 'ETHUSD':0.05, 'BCHUSD':0.05}
    symbol = df.iloc[0].at['symbol']
    
    df['std_bidSize'] /= df['mean_bidSize']
    df['std_askSize'] /= df['mean_askSize']
    
    # Sell -> -1, Buy -> 1, nan -> 0
    df['last_side'] = [0 if pd.isna(x) else 1 if x=='Buy' else -1 for x in df['last_side']]
    
    df['last_size'] = df['last_size'].fillna(0)
    df['std_size'] = df['last_size'].fillna(0)
    
    df['std_price'] = df['std_price'].fillna(0) / ticksize[symbol]
    
    df['MicroPriceAdjustment'] /= ticksize[symbol]
    
    return df

def normalize_cols(df):
    # TODO: add normalization for each column
    
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

# Calculate predicted y variables
xbt = y_vars(xbt)
eth = y_vars(eth)
bch = y_vars(bch)

# Calculate relative bid-ask spread
xbt = relative_bid_ask_spread(xbt)
eth = relative_bid_ask_spread(eth)
bch = relative_bid_ask_spread(bch)

# Calculate relative buys
xbt = relative_buys(xbt)
eth = relative_buys(eth)
bch = relative_buys(bch)

# Add sales dummy (1 if there was sales during past interval)
xbt = sales_dummy(xbt)
eth = sales_dummy(eth)
bch = sales_dummy(bch)

# Calculate relative moving averages
xbt = relative_smas(xbt)
eth = relative_smas(eth)
bch = relative_smas(bch)

# Clean column values
xbt = clean_columns(xbt)
eth = clean_columns(eth)
bch = clean_columns(bch)

# Normalize columns
xbt = normalize_cols(xbt)
eth = normalize_cols(eth)
bch = normalize_cols(bch)

# HISTOGRAMS

# for i, col in enumerate(bch.columns):
#     if i > 1:
#         plt.figure(i)
#         plt.hist(np.log(bch[col].dropna()+1), bins=30)
#         plt.title("log_"+col)
#         plt.show()

# plt.figure()
# plt.hist((bch['MicroPriceAdjustment']-bch['MicroPriceAdjustment'].mean())/np.std(bch['MicroPriceAdjustment']), bins=30)
# plt.title('MicroPriceAdjustement')
# plt.show()   

# a = 'BidPriceSMA_l'
# plt.hist(xbt[a], bins=100)
# plt.show()

# plt.hist(np.log(xbt[a]), bins=100)
# plt.show()

# pd.qcut(xbt[a], q=[0,.10,.5,.90,1]).value_counts()

# HEATMAP

cor = xbt.corr()
plt.figure(0, figsize=(9,4))
sns.set(font_scale=0.4)
hm = sns.heatmap(cor, cmap='RdBu_r', linewidth=0.5, vmin=-1, vmax=1, annot=True, fmt='.2f')
plt.title('Heatmap of indicator correlations', fontsize=10)
plt.savefig('heatmap', dpi=400, bbox_inches='tight')
plt.show()
plotdefaults()

# PREDICTIONS

# Get X columns from file
file3 = 'x_values.xlsx'
sheet = 'X_indicators'
x_columns = pd.read_excel(file3, sheet_name=sheet, header=None)
x_columns = x_columns[0].tolist()

# Get y columns
y_columns = [col for col in xbt.columns if col[:2]=='y_']

# Get X and y (NOTE: here only xbt, add others later)
X = xbt[x_columns]
y = xbt[y_columns]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
 

# Linear Regression

# Predict change in last best bid price with the micro price adjustment

# y = bch['last_bidPrice']
# y = (y.shift(1) - y)[1:]
# x = bch['MicroPriceAdjustment'][1:]
# x[x.between(-0.05,0.05)] = 0
# x = sm.add_constant(x, has_constant='add')
# model = sm.OLS(y,x).fit()
# model.summary()
