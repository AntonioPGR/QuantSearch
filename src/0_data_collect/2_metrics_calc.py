import json
import pandas as pd
import pandas_ta_classic as ta

tickers = []
try:
  with open("_SETTINGS/indexes.json", "r") as f:
    tickers = json.load(f)
except FileNotFoundError:
  print("Arquivo 'StocksList.json' não encontrado. Certifique-se de que o arquivo existe e contém os tickers desejados.")

for ticker in tickers:
    df = pd.read_csv(f"_DATA/all/{ticker.replace('.SA', '.csv')}")

    break


