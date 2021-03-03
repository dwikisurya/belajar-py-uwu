import quandl
import pandas as pd
import numpy as np
import sklearn 
import preprocessing, cross_validation, svm from sklearn.linear_model
import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

# Define forecast collumn
foreceast_col = 'Adj. Close'

# jika data = NaN maka di isi -9999
df.fillna(value=-99999, inplace=True)

# forecast dari data
forecast_out = int(math.ceil(0.01 * len(df)))

# beri nama label
df['label'] = df[forecast_col].shift(-forecast_out)

