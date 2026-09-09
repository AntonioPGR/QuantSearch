from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import entropy


class NetworkFeatureExtractor:
  FEATURE_NAMES = ["network_density", "avg_clustering", "component_count", "giant_component_ratio", "avg_path_length", "avg_degree", "std_degree", "avg_node_betweenness", "avg_node_closeness", "avg_node_eigenvector", "avg_edge_betweenness", "avg_jaccard_coef", "avg_adamic_adar", "avg_pref_attachment", "community_integration_ratio", "degree_entropy", ]

  @classmethod
  def extract(cls, adjacency_matrix: list[list[float]], ) -> (dict[str, float], list[float]):
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=float, )

    if adjacency_matrix.ndim != 2:
      raise ValueError("adjacency_matrix must be a 2D list")

    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
      raise ValueError("adjacency_matrix must be square")

    # Convert weighted/signed adjacency matrix
    # into a binary adjacency matrix.
    binary_adjacency = (adjacency_matrix != 0).astype(int)

    graph = nx.from_numpy_array(binary_adjacency)

    if graph.number_of_nodes() == 0:
      return cls._empty_features()

    if graph.number_of_edges() == 0:
      return cls._empty_features()

    number_of_nodes = graph.number_of_nodes()

    density = nx.density(graph)

    avg_clustering = nx.average_clustering(graph)

    components = list(nx.connected_components(graph))

    component_count = len(components)

    giant_component = max(components, key=len, )

    giant_component_ratio = (len(giant_component) / number_of_nodes)

    giant_graph = graph.subgraph(giant_component)

    if len(giant_graph) > 1:
      avg_path_length = (nx.average_shortest_path_length(giant_graph))
    else:
      avg_path_length = 0.0

    degrees = np.array([degree for _, degree in graph.degree()])

    avg_degree = float(np.mean(degrees))

    std_degree = float(np.std(degrees))

    node_betweenness = np.array(list(nx.betweenness_centrality(graph).values()))

    node_closeness = np.array(list(nx.closeness_centrality(graph).values()))

    avg_node_betweenness = float(np.mean(node_betweenness))

    avg_node_closeness = float(np.mean(node_closeness))

    try:
      node_eigenvector = np.array(list(nx.eigenvector_centrality(graph, max_iter=1000, ).values()))

      avg_node_eigenvector = float(np.mean(node_eigenvector))

    except nx.PowerIterationFailedConvergence:
      avg_node_eigenvector = 0.0

    avg_edge_betweenness = (cls._average_edge_betweenness(graph))

    link_features = cls._link_features(graph, avg_degree, )

    community_integration = (cls._community_integration(graph))

    degree_entropy = cls._degree_entropy(degrees)

    features = {"network_density": density, "avg_clustering": avg_clustering, "component_count": component_count, "giant_component_ratio": giant_component_ratio, "avg_path_length": avg_path_length, "avg_degree": avg_degree, "std_degree": std_degree, "avg_node_betweenness": avg_node_betweenness, "avg_node_closeness": avg_node_closeness, "avg_node_eigenvector": avg_node_eigenvector, "avg_edge_betweenness": avg_edge_betweenness, **link_features, "community_integration_ratio": community_integration, "degree_entropy": degree_entropy, }
    features = cls._normalize_features(features, number_of_nodes, )

    feature_vector = [float(features[name]) for name in cls.FEATURE_NAMES]

    return features, feature_vector

  @classmethod
  def _empty_features(cls, ) -> dict[str, float]:
    return {feature: 0.0 for feature in cls.FEATURE_NAMES}

  @staticmethod
  def _average_edge_betweenness(graph: nx.Graph, ) -> float:
    values = (nx.edge_betweenness_centrality(graph).values())

    return (float(np.mean(list(values))) if values else 0.0)

  @staticmethod
  def _link_features(graph: nx.Graph, avg_degree: float, ) -> dict[str, float]:
    non_edges = list(nx.non_edges(graph))

    if not non_edges:
      return {"avg_jaccard_coef": 1.0, "avg_adamic_adar": 0.0, "avg_pref_attachment": avg_degree ** 2, }

    jaccard_values = [value for _, _, value in nx.jaccard_coefficient(graph, non_edges, )]

    preferential_values = [value for _, _, value in nx.preferential_attachment(graph, non_edges, )]

    try:
      adamic_values = [value for _, _, value in nx.adamic_adar_index(graph, non_edges, )]

      avg_adamic_adar = float(np.mean(adamic_values))

    except ZeroDivisionError:
      avg_adamic_adar = 0.0

    return {"avg_jaccard_coef": float(np.mean(jaccard_values)), "avg_adamic_adar": avg_adamic_adar, "avg_pref_attachment": float(np.mean(preferential_values)), }

  @staticmethod
  def _community_integration(graph: nx.Graph, ) -> float:
    try:
      from community import community_louvain

      partition = (community_louvain.best_partition(graph))

      nodes = list(graph.nodes())

      if len(nodes) < 2:
        return 0.0

      same_community = 0
      total_pairs = 0

      for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
          total_pairs += 1

          if (partition[nodes[i]] == partition[nodes[j]]):
            same_community += 1

      return (same_community / total_pairs)

    except ImportError:
      return 0.0

  @staticmethod
  def _degree_entropy(degrees: np.ndarray, ) -> float:
    degree_counts = np.bincount(degrees)

    probabilities = (degree_counts / degree_counts.sum())

    return float(entropy(probabilities, base=2, ))

  @staticmethod
  def _normalize_features(features: dict[str, float], number_of_nodes: int, ) -> dict[str, float]:
    max_degree = number_of_nodes - 1

    max_degree_entropy = (np.log2(number_of_nodes) if number_of_nodes > 1 else 1.0)

    max_avg_path_length = (number_of_nodes - 1)

    max_pref_attachment = (max_degree ** 2)

    max_component_count = number_of_nodes

    bounds = {"network_density": (0.0, 1.0), "avg_clustering": (0.0, 1.0), "component_count": (1.0, float(max_component_count),), "giant_component_ratio": (0.0, 1.0), "avg_path_length": (0.0, float(max_avg_path_length),), "avg_degree": (0.0, float(max_degree),), "std_degree": (0.0, float(max_degree / 2),), "avg_node_betweenness": (0.0, 1.0), "avg_node_closeness": (0.0, 1.0), "avg_node_eigenvector": (0.0, 1.0), "avg_edge_betweenness": (0.0, 1.0), "avg_jaccard_coef": (0.0, 1.0), "avg_adamic_adar": (0.0, np.log(max_degree) if max_degree > 1 else 1.0,), "avg_pref_attachment": (0.0, float(max_pref_attachment),), "community_integration_ratio": (0.0, 1.0), "degree_entropy": (0.0, float(max_degree_entropy),), }

    normalized = {}

    for name in NetworkFeatureExtractor.FEATURE_NAMES:
      value = features[name]
      minimum, maximum = bounds[name]

      if maximum == minimum:
        normalized[name] = 0.0
        continue

      normalized[name] = ((value - minimum) / (maximum - minimum))

    return normalized
