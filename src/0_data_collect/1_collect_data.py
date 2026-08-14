import json
import os
import yfinance as yf
import pandas as pd

# SETTINGS
INTERVAL = "all"
OUTPUT_DIR = f"src/_DATA/{INTERVAL}"
PERIOD = "max"

# TICKERS 
tickers = []
try:
  with open("src/0_data_collect/_SETTINGS/indexes.json", "r") as f:
    tickers = json.load(f)
except FileNotFoundError:
  print("Arquivo 'StocksList.json' não encontrado. Certifique-se de que o arquivo existe e contém os tickers desejados.")
  
  
# DOWNLOAD DATA
os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)  
  
for ticker in tickers:
  data = yf.download(
    ticker,
    interval=INTERVAL,
    period=PERIOD,
  )
  data.columns = data.columns.droplevel(1)
  data.columns.name = None
  data = data.reset_index()

  filepath = f"{OUTPUT_DIR}/{ticker.split('.')[0]}.csv"
  if os.path.exists(filepath):
    old_data = pd.read_csv(filepath)
    combined_data = pd.concat([old_data, data], ignore_index=True)

    if "Datetime" in combined_data.columns:
      combined_data["Datetime"] = combined_data["Datetime"].astype(str)
      combined_data.drop_duplicates(subset="Datetime", keep="last", inplace=True)
      combined_data.sort_values("Datetime", inplace=True)
    elif "Date" in combined_data.columns:
      combined_data["Date"] = combined_data["Date"].astype(str)
      combined_data["Date"] = pd.to_datetime(combined_data["Date"], errors="coerce", format="mixed").dt.normalize()
      combined_data.dropna(subset=["Date"], inplace=True)
      combined_data.drop_duplicates(subset="Date", keep="last", inplace=True)
      combined_data.sort_values("Date", inplace=True)

    combined_data.to_csv(filepath, index=False)
  else:
    data.to_csv(filepath, index=False)

