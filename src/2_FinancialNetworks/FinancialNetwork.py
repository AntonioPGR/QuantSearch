import numpy as np
import pandas as pd

def calculateDAG(correlation_mx:pd.DataFrame):
	mask = np.triu(np.ones(correlation_mx.shape, dtype=bool),k=1)
	values = correlation_mx.to_numpy()[mask]
	absolute_values = np.abs(values)
	length = correlation_mx.shape[0]
	n = int(length * (length - 1) / 8)
	nth_index = np.argpartition(absolute_values, -n)[-n]
	threshold = absolute_values[nth_index]
	graph = correlation_mx.where((correlation_mx.abs() >= threshold) & (correlation_mx != 1), 0.0)
	return graph

def calculateDTN(correlation_mx:pd.DataFrame, threshold:(float|int)=0.65):
	graph = correlation_mx.where(correlation_mx.abs() >= threshold, 0.0)
	graph = np.sign(graph)
	n = correlation_mx.shape[0]
	graph.iloc[range(n), range(n)] = 0.0
	return graph

def calculateMST(correlation_mx:pd.DataFrame):
	edges = []
	n =  correlation_mx.shape[0]
	stocks = correlation_mx.columns.values.tolist()
	for i in range(n):
		for j in range(i + 1, n):
			rho = correlation_mx.iloc[i, j]
			distance = np.sqrt(2 * (1 - rho))
			edges.append((distance, i, j, rho))
	edges.sort(key=lambda x: x[0])
	parent = list(range(n))
	rank = [0] * n
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
	mst = pd.DataFrame(0.0, index=stocks, columns=stocks)
	edges_added = 0
	for distance, i, j, rho in edges:
		if union(i, j):
			stock1 = stocks[i]
			stock2 = stocks[j]
			mst.loc[stock1, stock2] = np.sign(rho)
			mst.loc[stock2, stock1] = np.sign(rho)
			edges_added += 1
			if edges_added == n - 1:
				break
	
	return mst