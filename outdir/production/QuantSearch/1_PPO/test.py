from pathlib import Path

import pandas as pd
import os
from glob import glob

maxLen = 0
for path in glob("_DATA/all/*.csv"):
    filename = os.path.splitext(os.path.basename(path))[0]
    lenght = len(pd.read_csv(path))
    if lenght > maxLen:
        maxLen = lenght

stocks = []
for path in glob("_DATA/all/*.csv"):
    filename = os.path.splitext(os.path.basename(path))[0]
    lenght = len(pd.read_csv(path))
    if lenght == maxLen:
        stocks.append(Path(filename).stem)

print(maxLen)
print(stocks)
