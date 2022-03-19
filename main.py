import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file1 = '../quote_20220318.csv'
file2 = '../trade_20220318.csv'

quote = pd.read_csv(file1, sep=',', index_col=0)
trade = pd.read_csv(file2, sep=',', index_col=0)

quote.index = pd.to_datetime(quote.index, format="%Y%m")


factors = factors.apply(lambda x: x.str.strip())
factors = factors.apply(pd.to_numeric)
factors