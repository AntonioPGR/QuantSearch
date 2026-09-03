from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy


class NetworkFeatureExtractor:
	"""Extract global features from an adjacency-matrix CSV file."""
	
	FEATURE_NAMES = [
		"network_density",
		"avg_clustering",
		"component_count",
		"giant_component_ratio",
		"avg_path_length",
		"avg_degree",
		"std_degree",
		"avg_node_betweenness",
		"avg_node_closeness",
		"avg_node_eigenvector",
		"avg_edge_betweenness",
		"avg_jaccard_coef",
		"avg_adamic_adar",
		"avg_pref_attachment",
		"community_integration_ratio",
		"degree_entropy",
	]
	
	@classmethod
	def extract_from_csv(
			cls,
			adjacency_matrix_path: str | Path,
	) -> dict[str, float]:
		"""Read an adjacency matrix from CSV and extract features."""
		
		adjacency_matrix = pd.read_csv(
			adjacency_matrix_path,
			index_col=0,
		)
		
		return cls.extract(adjacency_matrix)
	
	@classmethod
	def extract(
			cls,
			adjacency_matrix: pd.DataFrame,
	) -> dict[str, float]:
		"""Extract global features from an adjacency matrix."""
		
		binary_adjacency = adjacency_matrix.map(
			lambda value: int(value != 0)
		)
		
		graph = nx.from_pandas_adjacency(binary_adjacency)
		
		if graph.number_of_nodes() == 0:
			return cls._empty_features()
		
		if graph.number_of_edges() == 0:
			return cls._empty_features()
		
		number_of_nodes = graph.number_of_nodes()
		
		density = nx.density(graph)
		avg_clustering = nx.average_clustering(graph)
		
		components = list(nx.connected_components(graph))
		component_count = len(components)
		
		giant_component = max(components, key=len)
		giant_component_ratio = (
				len(giant_component) / number_of_nodes
		)
		
		giant_graph = graph.subgraph(giant_component)
		
		if len(giant_graph) > 1:
			avg_path_length = nx.average_shortest_path_length(
				giant_graph
			)
		else:
			avg_path_length = 0.0
		
		degrees = np.array(
			[degree for _, degree in graph.degree()]
		)
		
		avg_degree = float(np.mean(degrees))
		std_degree = float(np.std(degrees))
		
		node_betweenness = np.array(
			list(
				nx.betweenness_centrality(graph).values()
			)
		)
		
		node_closeness = np.array(
			list(
				nx.closeness_centrality(graph).values()
			)
		)
		
		avg_node_betweenness = float(
			np.mean(node_betweenness)
		)
		
		avg_node_closeness = float(
			np.mean(node_closeness)
		)
		
		try:
			node_eigenvector = np.array(
				list(
					nx.eigenvector_centrality(
						graph,
						max_iter=1000,
					).values()
				)
			)
			
			avg_node_eigenvector = float(
				np.mean(node_eigenvector)
			)
		
		except nx.PowerIterationFailedConvergence:
			avg_node_eigenvector = 0.0
		
		avg_edge_betweenness = cls._average_edge_betweenness(
			graph
		)
		
		link_features = cls._link_features(
			graph,
			avg_degree,
		)
		
		community_integration = cls._community_integration(
			graph
		)
		
		degree_entropy = cls._degree_entropy(degrees)
		
		return {
			"network_density": density,
			"avg_clustering": avg_clustering,
			"component_count": component_count,
			"giant_component_ratio": giant_component_ratio,
			"avg_path_length": avg_path_length,
			"avg_degree": avg_degree,
			"std_degree": std_degree,
			"avg_node_betweenness": avg_node_betweenness,
			"avg_node_closeness": avg_node_closeness,
			"avg_node_eigenvector": avg_node_eigenvector,
			"avg_edge_betweenness": avg_edge_betweenness,
			**link_features,
			"community_integration_ratio": community_integration,
			"degree_entropy": degree_entropy,
		}
	
	@classmethod
	def _empty_features(cls) -> dict[str, float]:
		return {
			feature: 0.0
			for feature in cls.FEATURE_NAMES
		}
	
	@staticmethod
	def _average_edge_betweenness(
			graph: nx.Graph,
	) -> float:
		values = nx.edge_betweenness_centrality(graph).values()
		
		return float(np.mean(list(values))) if values else 0.0
	
	@staticmethod
	def _link_features(
			graph: nx.Graph,
			avg_degree: float,
	) -> dict[str, float]:
		non_edges = list(nx.non_edges(graph))
		
		if not non_edges:
			return {
				"avg_jaccard_coef": 1.0,
				"avg_adamic_adar": 0.0,
				"avg_pref_attachment": avg_degree ** 2,
			}
		
		jaccard_values = [
			value
			for _, _, value in nx.jaccard_coefficient(
				graph,
				non_edges,
			)
		]
		
		preferential_values = [
			value
			for _, _, value in nx.preferential_attachment(
				graph,
				non_edges,
			)
		]
		
		try:
			adamic_values = [
				value
				for _, _, value in nx.adamic_adar_index(
					graph,
					non_edges,
				)
			]
			
			avg_adamic_adar = float(
				np.mean(adamic_values)
			)
		
		except ZeroDivisionError:
			avg_adamic_adar = 0.0
		
		return {
			"avg_jaccard_coef": float(
				np.mean(jaccard_values)
			),
			"avg_adamic_adar": avg_adamic_adar,
			"avg_pref_attachment": float(
				np.mean(preferential_values)
			),
		}
	
	@staticmethod
	def _community_integration(graph: nx.Graph) -> float:
		try:
			from community import community_louvain
			
			partition = community_louvain.best_partition(graph)
			nodes = list(graph.nodes())
			
			if len(nodes) < 2:
				return 0.0
			
			same_community = 0
			total_pairs = 0
			
			for i in range(len(nodes)):
				for j in range(i + 1, len(nodes)):
					total_pairs += 1
					
					if partition[nodes[i]] == partition[nodes[j]]:
						same_community += 1
			
			return same_community / total_pairs
		
		except ImportError:
			return 0.0
	
	@staticmethod
	def _degree_entropy(degrees: np.ndarray) -> float:
		degree_counts = np.bincount(degrees)
		probabilities = degree_counts / degree_counts.sum()
		
		return float(entropy(probabilities, base=2))
