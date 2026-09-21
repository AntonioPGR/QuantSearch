from pathlib import Path

import pandas as pd
from glob import glob

features_dfs = []
closes_dfs = []
dates = None
for path in glob("_DATA/featured/*.csv"):
    stem = Path(path).stem
    df = pd.read_csv(path).add_prefix(f"{stem}_")
    dates = df[f"{stem}_Date"].to_list()
    closes_dfs.append(df[f"{stem}_Close"])
    df.drop([f"{stem}_Date", f"{stem}_Open", f"{stem}_High", f"{stem}_Low", f"{stem}_Close", f"{stem}_Volume"], axis=1, inplace=True)
    features_dfs.append(df)

features_df = pd.concat(features_dfs, axis=1)
closes_df = pd.concat(closes_dfs, axis=1)
print(closes_df)
print(features_df)