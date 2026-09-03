import numpy as np
import pandas as pd


class NetworkBuilder:
	"""Create networks from a correlation matrix."""
	
	@staticmethod
	def calculate_dag(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
		"""
		Create a graph containing the strongest correlations.

		This keeps approximately the top quarter of all possible
		undirected edges, based on absolute correlation.
		"""
		
		if correlation_matrix.shape[0] < 2:
			return correlation_matrix.copy()
		
		mask = np.triu(
			np.ones(correlation_matrix.shape, dtype=bool),
			k=1,
		)
		
		values = correlation_matrix.to_numpy()[mask]
		absolute_values = np.abs(values)
		
		edge_count = len(values)
		number_to_keep = max(1, edge_count // 4)
		
		threshold_index = np.argpartition(
			absolute_values,
			-number_to_keep,
		)[-number_to_keep:]
		
		threshold = absolute_values[threshold_index].min()
		
		graph = correlation_matrix.where(
			correlation_matrix.abs() >= threshold,
			0.0,
		)
		
		graph.iloc[
			np.diag_indices_from(graph)
		] = 0.0
		
		return graph
	
	@staticmethod
	def calculate_dtn(
			correlation_matrix: pd.DataFrame,
			threshold: float = 0.65,
	) -> pd.DataFrame:
		"""Create a signed threshold network."""
		
		graph = correlation_matrix.where(
			correlation_matrix.abs() >= threshold,
			0.0,
		)
		
		graph = np.sign(graph)
		
		np.fill_diagonal(graph.values, 0.0)
		
		return graph
	
	@staticmethod
	def calculate_mst(
			correlation_matrix: pd.DataFrame,
	) -> pd.DataFrame:
		"""
		Create a minimum spanning tree using correlation distance:

				distance = sqrt(2 * (1 - correlation))
		"""
		
		stocks = correlation_matrix.columns.tolist()
		number_of_nodes = len(stocks)
		
		edges = []
		
		for i in range(number_of_nodes):
			for j in range(i + 1, number_of_nodes):
				correlation = correlation_matrix.iloc[i, j]
				distance = np.sqrt(2 * (1 - correlation))
				
				edges.append(
					(distance, i, j, correlation)
				)
		
		edges.sort(key=lambda edge: edge[0])
		
		parent = list(range(number_of_nodes))
		rank = [0] * number_of_nodes
		
		def find(node: int) -> int:
			if parent[node] != node:
				parent[node] = find(parent[node])
			
			return parent[node]
		
		def union(node_a: int, node_b: int) -> bool:
			root_a = find(node_a)
			root_b = find(node_b)
			
			if root_a == root_b:
				return False
			
			if rank[root_a] < rank[root_b]:
				parent[root_a] = root_b
			elif rank[root_a] > rank[root_b]:
				parent[root_b] = root_a
			else:
				parent[root_b] = root_a
				rank[root_a] += 1
			
			return True
		
		mst = pd.DataFrame(
			0.0,
			index=stocks,
			columns=stocks,
		)
		
		edges_added = 0
		
		for _, i, j, correlation in edges:
			if union(i, j):
				stock_a = stocks[i]
				stock_b = stocks[j]
				sign = np.sign(correlation)
				
				mst.loc[stock_a, stock_b] = sign
				mst.loc[stock_b, stock_a] = sign
				
				edges_added += 1
				
				if edges_added == number_of_nodes - 1:
					break
		
		return mst
