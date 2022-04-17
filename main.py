import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from matplotlib import rc_file_defaults as plotdefaults
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.inspection import permutation_importance
import shap
from warnings import simplefilter
from sklearn.feature_selection import SequentialFeatureSelector
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix

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
    
    df['mean_size'] = df['mean_size'].fillna(0)
    df['last_size'] = df['last_size'].fillna(0)
    df['std_size'] = df['last_size'].fillna(0)
    
    df['std_price'] = df['std_price'].fillna(0) / ticksize[symbol]
    
    df['MicroPriceAdjustment'] /= ticksize[symbol]
    
    return df

def normalize_cols(df):
    
    """
    Transforms features of df with log x, log(1+x), bins, and normalization methods

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with untransformed features

    Returns
    -------
    df : pandas.DataFrame
        Transformed DataFrame

    """
    
    # Instantiate transformers
    pt = PowerTransformer()
    ss = StandardScaler()
    
    # Create lists for features in each transformation group
    power_tf = ['mean_bidSize', 'mean_askSize', 'BidSizeSMA', 'AskSizeSMA', 'last_bidSize', 'last_askSize', 'std_bidSize', 'std_askSize', 'buys', 'buyVolume', 'sells', 'sellVolume', 'last_size', 'mean_size', 'std_price']
    std_tf = ['MicroPriceAdjustment', 'BidPriceSMA_s', 'AskPriceSMA_s', 'BidPriceSMA_l', 'AskPriceSMA_l', 'RelativeSpread','BuyPressure', 'RelativeBuys']
    
    # Transform features
    df[power_tf] = pt.fit_transform(df[power_tf])
    df['RelativeSpread'] = pd.qcut(df['RelativeSpread'], q=[0,.10,.5,.90,1], labels=False) # bins
    df[std_tf] = ss.fit_transform(df[std_tf])
    
    # Return transformed df
    return df

def heatmap(df):
    """
    Plots heatmap of features correlations in df

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of features

    Returns
    -------
    None.

    """
    
    cor = df.corr()
    plt.figure(0, figsize=(9,4))
    sns.set(font_scale=0.4)
    hm = sns.heatmap(cor, cmap='RdBu_r', linewidth=0.5, vmin=-1, vmax=1, annot=True, fmt='.2f')
    plt.title('Heatmap of indicator correlations', fontsize=10)
    plt.savefig('heatmap', dpi=400, bbox_inches='tight')
    plt.show()
    plotdefaults()
    
def histograms(df, cols):
    """
    Histograms of df[cols]

    Parameters
    ----------
    df : pd df
    cols : list

    Returns
    -------
    None.

    """
    for i, col in enumerate(df[cols]):
        plt.figure(i)
        plt.hist(eth[col], bins=30)
        plt.title(col)
        plt.show()
    
def add_vars(df):
    # Calculate micro price adjustments
    df = micro_price_adjustment(df)
    
    # Calculate predicted y variables
    df = y_vars(df)
    
    # Calculate relative bid-ask spread
    df = relative_bid_ask_spread(df)
    
    # Calculate relative buys
    df = relative_buys(df)
    
    # Add sales dummy (1 if there was sales during past interval)
    df = sales_dummy(df)
    
    # Calculate relative moving averages
    df = relative_smas(df)
    
    return df

def linear_regression(X_train, X_test, y_train, y_test, variables):
    # Initialize dataframes
    df_score = pd.DataFrame(index=['linear regression'], columns=[variables])
    df_mse = pd.DataFrame(index=['linear regression'], columns=[variables])
    
    for var in variables:
        # Run regression for each variable
        reg = LinearRegression().fit(X_train, y_train[var])
        y_pred = reg.predict(X_test)
        
        # Add results to dataframe
        df_score[var] = reg.score(X_test, y_test[var])
        df_mse[var] = mean_squared_error(y_test[var], y_pred)
    
    return df_score, df_mse

@ignore_warnings(category=ConvergenceWarning)
def lasso_regression(X_train, X_test, y_train, y_test, variables):
    # Initialize dataframes
    df_score = pd.DataFrame(index=['lasso regression'], columns=[variables])
    df_mse = pd.DataFrame(index=['lasso regression'], columns=[variables])
    
    for var in variables:
        # Find alpha with cross-validation, list potential alphas
        ls = list(np.arange(0.01, 3, 0.01)) + list(np.arange(3, 20, 1)) + list(np.arange(20, 250, 5)) + list(np.arange(250, 5000, 50)) + list(np.arange(5000, 10000, 100))
        
        # Run regression for each variable
        reg = LassoCV(cv=10, random_state=0, alphas=ls).fit(X_train, y_train[var])
        y_pred = reg.predict(X_test)
        
        # Add results to dataframe
        df_score[var] = reg.score(X_test, y_test[var])
        df_mse[var] = mean_squared_error(y_test[var], y_pred)
        
    return df_score, df_mse

def random_forest_regression(X_train, X_test, y_train, y_test, variables): 
    # Initialize dataframes
    df_score = pd.DataFrame(index=['random forest regression'], columns=[variables])
    df_mse = pd.DataFrame(index=['random forest regression'], columns=[variables])
    
    for var in variables:
        # Run regression for each variable
        reg = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=50, random_state=0).fit(X_train, y_train[var])
        y_pred = reg.predict(X_test)
        
        # Add results to dataframe
        df_score[var] = reg.score(X_test, y_test[var])
        df_mse[var] = mean_squared_error(y_test[var], y_pred)
    
    return df_score, df_mse

def xgb_regression(X_train, X_test, y_train, y_test, variables): 
    # Initialize dataframes
    df_score = pd.DataFrame(index=['XGBoost regeression'], columns=[variables])
    df_mse = pd.DataFrame(index=['XGBoost regeression'], columns=[variables])
    
    for var in variables:
        # Run regression for each variable
        reg = XGBRegressor(n_estimators=500, max_depth=2, eta=0.01, subsample=1, colsample_bytree=1, random_state=0).fit(X_train, y_train[var])
        y_pred = reg.predict(X_test)
        
        # Add results to dataframe
        df_score[var] = reg.score(X_test, y_test[var])
        df_mse[var] = mean_squared_error(y_test[var], y_pred)

    return df_score, df_mse

def logit_regression(X_train, X_test, y_train, y_test, variables):
    # Initialize dataframes
    df_score = pd.DataFrame(index=['logistic regression'], columns=[variables])
    df_mse = pd.DataFrame(index=['logistic regression'], columns=[variables])
    reports = []
    
    for var in variables:
        # Run regression for each variable
        reg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=10).fit(X_train, y_train[var])
        y_pred = reg.predict(X_test)
        # probabilities = reg.predict_proba(X_test)[:,1]
        # Add results to dataframe
        df_score[var] = reg.score(X_test, y_test[var])
        df_mse[var] = mean_squared_error(y_test[var], y_pred) # TODO: Niko, käykö järkeen käyttää y_pred vai probabilities?
        reports.append(classification_report(y_test[var], y_pred))
    
    return df_score, df_mse, reports

def xgb_importances(X, y):
    # Run regression
    model = XGBRegressor(n_estimators=500, max_depth=2, eta=0.01, subsample=1, colsample_bytree=1, random_state=0).fit(X, y)
    
    # Standard feature importance
    plt.figure(1)
    plot_importance(model, title=f'Feature importance: {y.name}')
    # list(zip(X.columns, model.feature_importances_))
    plt.show()
    
    # Permutation importance
    plt.figure(2)
    perm_imp = permutation_importance(model, X, y)
    sorted_idx = abs(perm_imp.importances_mean).argsort()
    plt.barh(X.columns[sorted_idx], abs(perm_imp.importances_mean[sorted_idx]))
    plt.show()
    
    # SHAP (Shapley values from game theory) importances
    plt.figure(3)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values)
    shap.plots.bar(shap_values)
    plt.show()
    
    return True

def linlog_importances(X, y, weight, method):
    weights = {'xbt':2, 'eth':2, 'bch':1}
    
    # Run regression
    if method == 'linear':
        reg = LinearRegression().fit(X, y)
    elif method == 'logistic':
        reg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=100).fit(X, y)
        
    # Use forward selection
    sfs = SequentialFeatureSelector(reg, n_features_to_select=5)
    sfs.fit(X, y)
    
    l = sfs.get_feature_names_out().tolist() * weights[weight]
    return l

def xgb_tuning(X, y, niter=10):
    # Define parameters to be tested
    params = {
        'n_estimators':[100,250,500],
        'max_depth':[2,3,4],
        'eta':[0.01,0.05,0.1,0.2],
        'subsample':[0.8,1.0],
        'colsample_bytree':[0.8,1.0]
        }
    
    # Base regression
    reg = XGBRegressor(n_estimators=100, max_depth=2, eta=0.01, subsample=0.6, colsample_bytree=0.6)
    
    # Randomized tuning search
    rscv = RandomizedSearchCV(reg, params, n_iter=niter)
    rscv.fit(X, y)
    
    return rscv.best_params_

# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Read files
file1 = '../quote_20220318.csv'
file2 = '../trade_20220318.csv'
file3 = '../quote_20220320.csv'
file4 = '../trade_20220320.csv'

quote = pd.read_csv(file1, sep=',', index_col=0)
trade = pd.read_csv(file2, sep=',', index_col=0)
test_quote = pd.read_csv(file3, sep=',', index_col=0)
test_trade = pd.read_csv(file4, sep=',', index_col=0)

# Set datetime index
quote.index = pd.to_datetime(quote.index, format="%Y-%m-%dD%H:%M:%S.%f")
trade.index = pd.to_datetime(trade.index, format="%Y-%m-%dD%H:%M:%S.%f")
test_quote.index = pd.to_datetime(test_quote.index, format="%Y-%m-%dD%H:%M:%S.%f")
test_trade.index = pd.to_datetime(test_trade.index, format="%Y-%m-%dD%H:%M:%S.%f")

# Separate into three dataframes based on crypto
xbt_quote = quote[quote['symbol'] == 'XBTUSD']
eth_quote = quote[quote['symbol'] == 'ETHUSD']
bch_quote = quote[quote['symbol'] == 'BCHUSD']

xbt_trade = trade[trade['symbol'] == 'XBTUSD']
eth_trade = trade[trade['symbol'] == 'ETHUSD']
bch_trade = trade[trade['symbol'] == 'BCHUSD']

test_xbt_quote = test_quote[test_quote['symbol'] == 'XBTUSD']
test_eth_quote = test_quote[test_quote['symbol'] == 'ETHUSD']
test_bch_quote = test_quote[test_quote['symbol'] == 'BCHUSD']

test_xbt_trade = test_trade[test_trade['symbol'] == 'XBTUSD']
test_eth_trade = test_trade[test_trade['symbol'] == 'ETHUSD']
test_bch_trade = test_trade[test_trade['symbol'] == 'BCHUSD']

# Resample data into (1s) intervals
ival = '1s'

# Resample data and do feature engineering
xbt = resample_df(xbt_quote, xbt_trade, ival)
eth = resample_df(eth_quote, eth_trade, ival)
bch = resample_df(bch_quote, bch_trade, ival)

test_xbt = resample_df(test_xbt_quote, test_xbt_trade, ival)
test_eth = resample_df(test_eth_quote, test_eth_trade, ival)
test_bch = resample_df(test_bch_quote, test_bch_trade, ival)

# Add relevant variables
xbt = add_vars(xbt)
eth = add_vars(eth)
bch = add_vars(bch)

test_xbt = add_vars(test_xbt)
test_eth = add_vars(test_eth)
test_bch = add_vars(test_bch)

# Clean column values
xbt = clean_columns(xbt)
eth = clean_columns(eth)
bch = clean_columns(bch)

test_xbt = clean_columns(test_xbt)
test_eth = clean_columns(test_eth)
test_bch = clean_columns(test_bch)

# Normalize columns
xbt = normalize_cols(xbt)
eth = normalize_cols(eth)
bch = normalize_cols(bch)

"""
test_xbt = normalize_cols(test_xbt)
test_eth = normalize_cols(test_eth)
test_bch = normalize_cols(test_bch)
"""

# Set names for dataframes
xbt.name = 'xbt'
eth.name = 'eth'
bch.name = 'bch'

test_xbt.name = 'test_xbt'
test_eth.name = 'test_eth'
test_bch.name = 'test_bch'

# Make predictions
# Get X columns from file
file3 = 'x_values.xlsx'
sheet = 'X_indicators'
x_columns = pd.read_excel(file3, sheet_name=sheet, header=None)
x_columns = x_columns[0].tolist()

# Get y columns
y_columns = [col for col in xbt.columns if col[:2]=='y_']

# Loop for all dataframes, and add results to master dataframes
scores = pd.DataFrame()
mses = pd.DataFrame()

"""
for df in [xbt, eth, bch]:
    # Get X and y values
    X = df[x_columns][:-1]
    y = df[y_columns][:-1]
    
    # Split data into train and test (validation set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    # Separate boolean regressions
    vars1 = y.columns.to_list()[0:2] + y.columns.to_list()[4:]
    vars2 = y.columns.to_list()[2:4]

    # Run regressions
    lr_score, lr_mse = linear_regression(X_train, X_test, y_train[vars1], y_test[vars1], vars1)
    la_score, la_mse = lasso_regression(X_train, X_test, y_train[vars1], y_test[vars1], vars1)
    rf_score, rf_mse = random_forest_regression(X_train, X_test, y_train[vars1], y_test[vars1], vars1)
    xgb_score, xgb_mse = xgb_regression(X_train, X_test, y_train[vars1], y_test[vars1], vars1)
    lt_score, lt_mse, lt_reports = logit_regression(X_train, X_test, y_train[vars2], y_test[vars2], vars2)

    # Concatenate values
    score = pd.concat([lr_score, la_score, rf_score, xgb_score, lt_score])
    mse = pd.concat([lr_mse, la_mse, rf_mse, xgb_mse, lt_mse])
    score.index = [df.name+" "+i for i in score.index]
    mse.index = [df.name+" "+i for i in mse.index]
    
    scores = pd.concat([scores, score])
    mses = pd.concat([mses, mse])
"""

# Based on above results, we choose the following models for y values
xgb_ylist = ['y_bidSize', 'y_askSize', 'y_sells', 'y_buys']
linear_ylist = ['y_changeBidPrice', 'y_changeAskPrice']
logit_ylist = ['y_bidTickDown', 'y_askTickUp']


# Draw importance graphs for xgb, importance selection done manually due to differences in approaches
# Can be commented out
"""
for col in xgb_ylist:
    for df in [xbt, eth, bch]:
        X = df[x_columns][:-1]
        y = df[y_columns][:-1]
        
        xgb_importances(X, y[col])

# Find most important features with forward selection
linear_selection = []
for col in linear_ylist:
    linear_feat_list = []
    
    for df in [xbt, eth, bch]:
        X = df[x_columns][:-1]
        y = df[y_columns][:-1]
        
        linear_feat_list.extend(linlog_importances(X, y[col], df.name, method='linear'))
    
    linear_selection.append([i[0] for i in Counter(linear_feat_list).most_common(5)])

# Find most important features with forward selection
logistic_selection = []
for col in logit_ylist:
    logistic_feat_list = []
    
    for df in [xbt, eth, bch]:
        X = df[x_columns][:-1]
        y = df[y_columns][:-1]
        
        logistic_feat_list.extend(linlog_importances(X, y[col], df.name, method='logistic'))
    
    logistic_selection.append([i[0] for i in Counter(logistic_feat_list).most_common(7)])
"""

# Based on previous results, we choose the following features
# y_changeBidPrice
features_y0 = ['last_bidSize', 'last_askSize', 'last_side', 'RelativeSpread', 'mean_bidSize'] #linear_selection[0]

# y_changeAskPrice
features_y1 = ['last_bidSize', 'last_askSize', 'last_side', 'BuyPressure', 'mean_bidSize'] #linear_selection[1]

# y_bidTickDown, NOTE: deleted 90%+ correlated features
features_y2 = ['buys', 'std_price', 'AskPriceSMA_s', 'SalesDummy', 'last_side'] #[i for i in logistic_selection[0] if i not in ['buyVolume', 'BidPriceSMA_s']]

# y_askTickUp, NOTE: deleted 90%+ correlated features
features_y3 = ['std_price', 'BidPriceSMA_s', 'sells', 'buys', 'last_size'] #[i for i in logistic_selection[1] if i not in ['AskPriceSMA_s', 'sellVolume']]

# y_bidSize
features_y4 = ['last_bidSize', 'BidSizeSMA', 'MicroPriceAdjustment', 'BuyPressure', 'last_askSize']

# y_askSize
features_y5 = ['last_askSize', 'AskSizeSMA', 'BuyPressure', 'last_bidSize', 'AskPriceSMA_l']

# y_sells
features_y6 = ['sells', 'BidPriceSMA_l', 'mean_bidSize', 'last_askSize', 'BidSizeSMA']

# y_buys
features_y7 = ['buys', 'last_askSize', 'BidPriceSMA_s', 'RelativeSpread', 'std_askSize']

lin_reg_features = {'y_changeBidPrice': features_y0, 'y_changeAskPrice': features_y1}
logit_reg_features = {'y_bidTickDown': features_y2, 'y_askTickUp': features_y3}
xgb_features = {'y_bidSize':features_y4, 'y_askSize':features_y5, 'y_sells':features_y6, 'y_buys':features_y7}


"""
# Parameter tuning for xgb-based models
Can be commented if using manual input below
best_params = []
for col in xgb_ylist:
    best_df = pd.DataFrame()
    
    for df in [xbt, eth, bch]:
        X = df[xgb_features[col]][:-1]
        y = df[y_columns][:-1]
        
        best = xgb_tuning(X, y[col], niter=30)
        print(best)
        
        b = pd.DataFrame(best.items()).set_index(0)
        best_df = best_df.merge(b, how='outer', left_index=True, right_index=True)
    
    best_params.append(best_df.transpose().median())
"""

# Manual input, avoid excessive runtime
params_index = ['subsample', 'n_estimators', 'max_depth', 'eta', 'colsample_bytree']
params_y4 = pd.Series(data=[0.8,250,2,0.05,0.8], index=params_index)
params_y5 = pd.Series(data=[1,250,3,0.01,1], index=params_index)
params_y6 = pd.Series(data=[0.8,100,3,0.05,1], index=params_index)
params_y7 = pd.Series(data=[1,500,2,0.01,1], index=params_index)
xgb_params = [params_y4, params_y5, params_y6, params_y7]

# Define variables for loops
dfs = [xbt, eth, bch] # All assets
all_ylist = xgb_ylist + linear_ylist + logit_ylist # All dependent vars
final_features = {**lin_reg_features, **logit_reg_features, **xgb_features} # Dict of selected independent vars for each dependent var
xgb_map = {'y_bidSize':xgb_params[0], 'y_askSize':xgb_params[1], 'y_sells':xgb_params[2], 'y_buys':xgb_params[3]} # map of var to hyperparameters

# Train final models
def final_models():
    """
    Fits final model for all dependent variables

    Returns
    -------
    results : resulting dataframe

    """
    
    # results dataframe
    results = pd.DataFrame(columns=['Variable', 'Asset', 'Model', 'Train R^2', 'Test R^2', 'Train MSE', 'Test MSE'])
    
    # Iterate over assets
    for df in dfs:
        # Iterate over dependent vars
        for y in all_ylist:
            # Do train-test split
            X_train = df[final_features[y]].head(-1)
            X_test = eval('test_'+df.name)[final_features[y]].head(-1)
            y_train = df[y].head(-1)
            y_test = eval('test_'+df.name)[y].head(-1)
            
            # Fit model depending on group
            if y in linear_ylist:
                model_name = 'Linear Regression'
                reg = LinearRegression().fit(X_train, y_train)
                train_score = reg.score(X_train, y_train)
                test_score = reg.score(X_test, y_test)
                y_pred_train = reg.predict(X_train)
                y_pred_test = reg.predict(X_test)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
            elif y in logit_ylist:
                model_name = 'Logistic Regression'
                reg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=30).fit(X_train, y_train)
                train_score = reg.score(X_train, y_train)
                test_score = reg.score(X_test, y_test)
                y_pred_train = reg.predict(X_train)
                y_pred_test = reg.predict(X_test)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
            else:
                model_name = 'XG Boost'
                reg = XGBRegressor(n_estimators=int(xgb_map[y]['n_estimators']), max_depth=int(xgb_map[y]['max_depth']), eta=xgb_map[y]['eta'], subsample=xgb_map[y]['subsample'], colsample_bytree=xgb_map[y]['colsample_bytree'], random_state=0)
                reg = reg.fit(X_train, y_train)
                train_score = reg.score(X_train, y_train)
                test_score = reg.score(X_test, y_test)
                y_pred_train = reg.predict(X_train)
                y_pred_test = reg.predict(X_test)
                if y in ['y_bidSize', 'y_askSize']:
                    y_pred_train = [i if i>100 else 100 for i in y_pred_train]
                    y_pred_test = [i if i>100 else 100 for i in y_pred_test]
                else: # sells or buys, which cannot be negative
                    y_pred_train = [i if i>=0 else 0 for i in y_pred_train]
                    y_pred_test = [i if i>=0 else 0 for i in y_pred_test]
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                
            # Append diagnostics to results
            results = results.append({'Variable': y, 'Asset': df.name, 'Model': model_name, 'Train R^2': train_score, 'Test R^2': test_score, 'Train MSE': mse_train, 'Test MSE': mse_test}, ignore_index=True)
    
    # Return results
    return results

# Compute final results and export to Excel
final_results = final_models()
final_results.to_excel("final_results.xlsx")

# Try linear regression for one of xbt y variables
X_train = xbt[final_features['y_changeBidPrice']].head(-1)
X_test = test_xbt[final_features['y_changeBidPrice']].head(-1)
y_train = xbt['y_changeBidPrice'].head(-1)
y_test = test_xbt['y_changeBidPrice'].head(-1)
reg = LinearRegression().fit(X_train, y_train)
y_pred_test = reg.predict(X_test)
y_pred_test = pd.DataFrame(y_pred_test)
y_pred_test.describe()


def make_classification_report(df, y):
    """
    Fits logistic regression variable y in dataframe df 

    Returns
    -------
    classification report of predictions

    """
    
    # Do train-test split
    X_train = df[final_features[y]].head(-1)
    X_test = eval('test_'+df.name)[final_features[y]].head(-1)
    y_train = df[y].head(-1)
    y_test = eval('test_'+df.name)[y].head(-1)
    # Fit model
    reg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=30).fit(X_train, y_train)
    # Predict
    y_pred = reg.predict(X_test)
    # Return resulting classification report
    return classification_report(y_test, y_pred)
 
# Print classification report for all classifiers   
for df in dfs:
    for y in logit_ylist:
        print(make_classification_report(df, y))


def plot_confusion_matrices():
    """
    Generates a plot of confusion matrices for logistic regression classifiers
    Saves the resulting figure

    Returns
    -------
    None.

    """

    clfs = [] # buffer for classifiers
    dfs = [xbt, eth, bch] # df container
    title_map = {} # dict map for subplot headers
    for df in dfs:
        for y in logit_ylist:
            X_train = df[final_features[y]].head(-1)
            X_test = eval('test_'+df.name)[final_features[y]].head(-1)
            y_train = df[y].head(-1)
            y_test = eval('test_'+df.name)[y].head(-1)
            clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=30).fit(X_train, y_train)
            clfs.append(clf)
            title_map[id(clf)] = f'{y} ({df.name})'

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
    for clf, ax in zip(clfs, axes.flatten()):
        print(clf, ax)
        plot_confusion_matrix(clf, 
                              X_test, 
                              y_test, 
                              ax=ax,
                              normalize='true',
                              cmap='Blues')
        ax.title.set_text(title_map[id(clf)])
    plt.tight_layout()
    plt.savefig('conf_mat.png')
    plt.show()      

# Plot confusion matrices for classifiers
plot_confusion_matrices()

def plot_correlation_matrix(df):
    """
    Generates heatmap of correlations between features of given df and saves the resulting figure

    Parameters
    ----------
    df : pandas DataFrame

    Returns
    -------
    None.

    """
    corr = df[x_columns].corr() # compute feature correlations
    plt.figure(figsize = (19,14))
    sns.heatmap(corr, cmap=sns.color_palette("viridis", as_cmap=True), annot=True, annot_kws={"size": 10}, fmt='.2f')
    plt.title(f'Heatmap of Feature Correlations ({df.name})', fontsize=18)
    plt.xticks(rotation=90) # set rotation of features on x axis 
    plt.savefig(f'corr_heatmap_{df.name}.png')

# Generate feature correlation heatmaps for all assets
for df in dfs:
    plot_correlation_matrix(df)
    

def plot_standardization_results(df, feats):
    """
    Plots before and after standardization subplots for feats

    Parameters
    ----------
    df : pd.DataFrame
    feats : list

    Returns
    -------
    None.

    """
    before = eval(f'test_{df.name}')
    after = df
    fig, axes = plt.subplots(3, 2, figsize=(12,12))
    for i, feat in enumerate(feats):
        sns.distplot(before[feat], ax=axes[i,0])
        sns.distplot(after[feat], ax=axes[i,1])
        axes[i,0].set_title(f'Raw Data ({feat})')
        axes[i,1].set_title(f'Standardized ({feat})')
    fig.suptitle("Features Before and After Standardization", fontsize=16)
    fig.tight_layout()
    plt.savefig("standardization_results")

# Plot standardization result for bidSize related features         
features = ['last_bidSize', 'mean_bidSize', 'BidSizeSMA']
plot_standardization_results(xbt, features)
        

def plot_pairs(sample_size=100):
    """
    Generates pairwise scatter plots for features in linear regression models
    Uses a random sample of the features to avoid excessive computing time

    Parameters
    ----------
    sample_size : int
        Size of the random sample that is drawn from the df. The default is 100.

    Returns
    -------
    None.

    """
    for df in dfs:    
        for y in linear_ylist:
            sample_df = df[final_features[y]].sample(sample_size, axis=0)
            g = sns.pairplot(sample_df, diag_kind="kde")
            g.map_lower(sns.kdeplot, levels=4, color=".2")
            g.fig.suptitle(f'{df.name}: {y}', fontsize=16)
            plt.tight_layout()
            plt.show()
 
# Plot pairwise scatter plots for all features used in linear regression models
plot_pairs()

# Individually chosen example plot from plot_pairs function
feature_df = xbt[final_features['y_changeAskPrice']].sample(1000, axis=0)
g = sns.pairplot(feature_df, diag_kind="kde")
g.map_lower(sns.kdeplot, levels=4, color=".2")
g.fig.suptitle('xbt: y_changeAskPrice', fontsize=16)
plt.tight_layout()
plt.savefig('xbt_feature_pairplot')
plt.show()


# TODO: If time, consider subsampling: can we predict large or smalle values better, etc.
# TODO: If time, See if reducing variables from 5 significantly worsens results