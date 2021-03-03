import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

# Inisiasi Dataframe wiki/.google
df = quandl.get("WIKI/GOOGL")

#define header dataframe
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# buat header baru hl_pct
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

# buat header baru pct_change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#Data frame 'baru' dengan tambahan 2 header baru diatas
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Define forecast collumn
forecast_col = 'Adj. Close'
# jika data NaN fillna berfungsi mereplace dengan value -99999
df.fillna(value=-99999, inplace=True)
# inisiasi var forecast dari data?
forecast_out = int(math.ceil(0.01 * len(df)))
# but collumn baru (We'll assume all current columns are our features)
df['label'] = df[forecast_col].shift(-forecast_out)

# X= features sedangkan y=label

# define dataframe untuk var X kecuali kolom label
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
# jika data NaN maka akan di drop/hapus?
df.dropna(inplace=True)
# define label dataframe
y = np.array(df['label'])

# The return here is the training set of features, testing set of features, training set of labels, and testing set of labels.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#define clasifiers
clf = LinearRegression(n_jobs=-1)
# Start train
# Here, we're "fitting" our training features and training labels.
clf.fit(X_train, y_train)

# Hasil 'train' 
confidence = clf.score(X_test, y_test)

#print out
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
