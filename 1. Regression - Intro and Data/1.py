import pandas as pd
import quandl

# Ambil data dari wikiprices | https://www.quandl.com/databases/WIKIP/documentation 
df = quandl.get("WIKI/GOOGL")

# Df dibawah merupakan dataframe
# Iris data(ambil collumn) dari var df diatas, ambil adj.open, adj.high, adj.low, adj.close , adj.volume
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Transofrm data 
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Membuat dataframe baru dari data diatas
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]
print(df.head())