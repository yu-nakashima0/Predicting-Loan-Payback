import pandas as pd


df = pd.read_csv('train.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)