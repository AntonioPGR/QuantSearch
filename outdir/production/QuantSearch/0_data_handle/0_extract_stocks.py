import pandas as pd
import json

df = pd.read_csv("src/0_data_handle/_SETTINGS/ibovespa_portfolio.csv", sep=";")
indexes = df["Código"].to_list()

for i in range(len(indexes)):
    indexes[i] += ".SA"
    
with open("src/0_data_handle/_SETTINGS/indexes.json", "w") as file:
    json.dump(indexes, file)