import pandas as pd

df = pd.read_csv("./train.csv")

print(df.columns)

for i in df.columns:
    if df[i].dtype == object:
        print(df[i].value_counts())
        print(df[i].unique())