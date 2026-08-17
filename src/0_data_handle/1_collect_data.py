import json
import os
import yfinance as yf
import pandas as pd

# TICKERS
tickers = []
try:
  with open("_SETTINGS/indexes.json", "r") as f:
    tickers = json.load(f)
except FileNotFoundError:
  print("Arquivo 'StocksList.json' não encontrado. Certifique-se de que o arquivo existe e contém os tickers desejados.")
  
  
# DOWNLOAD DATA
os.makedirs(f"_DATA/all", exist_ok=True)
  
for ticker in tickers:
  df = yf.download(
    ticker,
    interval="1d",
    period="max",
  )
  df.columns = df.columns.droplevel(1)
  df.columns.name = None
  df = df.reset_index()

  filepath = f"_DATA/all/{ticker.split('.')[0]}.csv"
  df.dropna(subset=df.columns.tolist(), inplace=True)
  df.to_csv(filepath, index=False)

