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
        probabilities = reg.predict_proba(X_test)[:,1]
        
        # Add results to dataframe
        df_score[var] = reg.score(X_test, y_test[var])
        df_mse[var] = mean_squared_error(y_test[var], y_pred) # TODO: Niko, käykö järkeen käyttää y_pred vai probabilities?
        reports.append(classification_report(y_test[var], y_pred))
    
    return df_score, df_mse, reports

def xgb_importances(X, y):
    # Run regression
    model = XGBRegressor(n_estimators=500, max_depth=2, eta=0.01, subsample=1, colsample_bytree=1, random_state=0).fit(X, y)
    
    # Standard feature importance
    plot_importance(model, title=f'Feature importance: {y.name}')
    # list(zip(X.columns, model.feature_importances_))
    
    # Permutation importance
    perm_imp = permutation_importance(model, X, y)
    sorted_idx = abs(perm_imp.importances_mean).argsort()
    plt.barh(X.columns[sorted_idx], abs(perm_imp.importances_mean[sorted_idx]))
    
    # SHAP (Shapley values from game theory) importances
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values)
    shap.plots.bar(shap_values)
    
    return True

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

# Add relevant variables
xbt = add_vars(xbt)
eth = add_vars(eth)
bch = add_vars(bch)

# Clean column values
xbt = clean_columns(xbt)
eth = clean_columns(eth)
bch = clean_columns(bch)

# Normalize columns
xbt = normalize_cols(xbt)
eth = normalize_cols(eth)
bch = normalize_cols(bch)

# Set names for dataframes
xbt.name = 'xbt'
eth.name = 'eth'
bch.name = 'bch'

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

# Based on above results, we choose the following models for y values
xgb_ylist = ['y_bidSize', 'y_askSize', 'y_sells', 'y_buys']
linear_ylist = ['y_changeBidPrice', 'y_changeAskPrice']
logit_ylist = ['y_bidTickDown', 'y_askTickUp']

# Draw importance graphs
for df in [xbt, eth, bch]:
    X = df[x_columns][:-1]
    y = df[y_columns][:-1]
    
    for col in xgb_ylist:
        xgb_importances(X, y[col])
    
    # for col in linear_ylist:
    #     linear_importances(X, y[col])
    
    # for col in logit_ylist:
    #     logit_importances(X, y[col])


# Based on previous results, we choose the following features
# y_bidSize
features_y4 = ['last_bidSize', 'BidSizeSMA', 'MicroPriceAdjustment', 'BuyPressure', 'last_askSize']

# y_askSize
features_y5 = ['last_askSize', 'AskSizeSMA', 'BuyPressure', 'last_bidSize', 'AskPriceSMA_l']

# y_sells
features_y6 = ['sells', 'sellVolume', 'BidPriceSMA_l', 'mean_bidSize', 'last_askSize']

# y_buys
features_y7 = ['buys', 'buyVolume', 'last_askSize', 'BidPriceSMA_s', 'RelativeSpread']




# TODO: Finding out out whether we should use probabilities or predictions in logit MSE
# TODO: Figuring out why logit regression is so slow
# TODO: Wrapper methods / feature importances for selected models (wrapper for linreg, figure out best ways for xgb and logit, target: 5-10 dep. vars)
# TODO: Bivariate Plots to see dependecies (pairs scatterplots with kde plots, similarly as in exercises)
# TODO: Final hyperparameter tunings (consider cross validation, grid search? Utilize sklearn pipelines&tools)
# TODO: Load completely new data and run models on that data, analyze&reflect results, show plots(?)




# histograms(eth, x_columns)
