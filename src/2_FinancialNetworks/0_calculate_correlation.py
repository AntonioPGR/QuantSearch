import os
from statistics import correlation

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

STOCKS = [
	"SBSP3", "BBDC3", "BRAP4", "VALE3", "GGBR4", "ISAE4",
	"CSNA3", "CMIG4", "EMBJ3", "CPLE3", "USIM5", "ITSA4",
	"AXIA3", "VIVT3", "WEGE3", "POMO4", "PETR4", "PETR3"
]
CORRELATIONS = ["pearsons", "spearmans", "kendalls"]

os.makedirs("_DATA/correlations", exist_ok=True)

closes = {
	stock: pd.read_csv(f"_DATA/all/{stock}.csv",index_col="Date")["Close"]
	for stock in STOCKS
}

correlations = {}
for cor in CORRELATIONS:
	correlations[cor] = pd.DataFrame(index=STOCKS, columns=STOCKS, dtype=float)

for stock in STOCKS:
	for stock2 in STOCKS:
		if stock == stock2:
			for cor in CORRELATIONS:
				correlations[cor].loc[stock, stock2] = 1.0
			continue
			
		x = closes[stock]
		y = closes[stock2]
		correlations["pearsons"].loc[stock, stock2] = round(pearsonr(x, y).statistic, 5)
		correlations["spearmans"].loc[stock, stock2] = round(spearmanr(x, y).statistic, 5)
		correlations["kendalls"].loc[stock, stock2] = round(kendalltau(x, y).statistic, 5)

for cor in CORRELATIONS:
	correlations[f"{cor}"].to_csv(f"2_FinancialNetworks/_DATA/correlations/{cor}.csv")

# DAG
# for cor in CORRELATIONS:
# 	mask = np.triu(np.ones(correlations[cor].shape, dtype=bool),k=1)
# 	values = correlations[cor].to_numpy()[mask]
# 	absolute_values = np.abs(values)
# 	length = len(STOCKS)
# 	n = int(length * (length - 1) / 8)
# 	nth_index = np.argpartition(absolute_values, -n)[-n]
# 	threshold = absolute_values[nth_index]
# 	graph = correlations[cor].where((correlations[cor].abs() >= threshold) & (correlations[cor] != 1), 0.0)
# 	graph.to_csv(f"2_FinancialNetworks/_DATA/correlations/{cor}_graph.csv")
	
# DTN
# for cor in CORRELATIONS:
# 	threshold = 0.65
# 	graph = correlations[cor].where(correlations[cor].abs() >= threshold, 0.0)
# 	graph = np.sign(graph)
# 	graph.iloc[range(len(STOCKS)), range(len(STOCKS))] = 0.0
# 	graph.to_csv(f"2_FinancialNetworks/_DATA/correlations/{cor}_graph.csv")

for cor in CORRELATIONS:
	correlation = correlations[cor]
	edges = []
	
	for i in range(len(STOCKS)):
		for j in range(i + 1, len(STOCKS)):
			rho = correlation.iloc[i, j]
			distance = np.sqrt(2 * (1 - rho))
			edges.append((distance, i, j, rho))
			
	edges.sort(key=lambda x: x[0])
	parent = list(range(len(STOCKS)))
	rank = [0] * len(STOCKS)
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(x, y):
		x = find(x)
		y = find(y)
		if x == y:
			return False
		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x
		else:
			parent[y] = x
			rank[x] += 1
		return True
	
	mst = pd.DataFrame(
		0.0,
		index=STOCKS,
		columns=STOCKS
	)
	
	edges_added = 0
	for distance, i, j, rho in edges:
		if union(i, j):
			stock1 = STOCKS[i]
			stock2 = STOCKS[j]
			mst.loc[stock1, stock2] = np.sign(rho)
			mst.loc[stock2, stock1] = np.sign(rho)
			edges_added += 1
			if edges_added == len(STOCKS) - 1:
				break
	
	mst.to_csv(f"2_FinancialNetworks/_DATA/correlations/{cor}_mst.csv")

