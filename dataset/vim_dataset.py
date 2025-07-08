import pandas as pd
df = pd.read_parquet("./valid.parquet")
print(df.head())
print(df.iloc[0].to_dict())