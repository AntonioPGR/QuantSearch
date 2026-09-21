import numpy as np


class NetworkBuilder:

  @staticmethod
  def _calculate_correlation_matrix(prices: list[list[float]], ) -> np.ndarray:
    prices_array = np.asarray(prices, dtype=float)

    if prices_array.ndim != 2:
      raise ValueError("prices must be a 2D list")

    if prices_array.shape[0] < 2:
      raise ValueError("At least two timesteps are required")

    # Rows = timesteps
    # Columns = stocks
    return np.corrcoef(prices_array, rowvar=False)

  @staticmethod
  def calculate_dag(prices: list[list[float]], ) -> np.ndarray:
    correlation_matrix = NetworkBuilder._calculate_correlation_matrix(prices)

    number_of_stocks = correlation_matrix.shape[0]

    if number_of_stocks < 2:
      return np.zeros_like(correlation_matrix)

    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1, )

    values = correlation_matrix[mask]
    absolute_values = np.abs(values)

    edge_count = len(values)
    number_to_keep = max(1, edge_count // 4)

    threshold_index = np.argpartition(absolute_values, -number_to_keep, )[-number_to_keep:]

    threshold = absolute_values[threshold_index].min()

    graph = np.where(np.abs(correlation_matrix) >= threshold, correlation_matrix, 0.0, )

    np.fill_diagonal(graph, 0.0)

    return graph

  @staticmethod
  def calculate_dtn(prices: list[list[float]], threshold: float = 0.65, ) -> np.ndarray:
    correlation_matrix = NetworkBuilder._calculate_correlation_matrix(prices)
    graph = np.where(np.abs(correlation_matrix) >= threshold, np.sign(correlation_matrix), 0.0, )
    np.fill_diagonal(graph, 0.0)
    return graph

  @staticmethod
  def calculate_mst(prices: list[list[float]], ) -> np.ndarray:
    correlation_matrix = NetworkBuilder._calculate_correlation_matrix(prices)
    number_of_stocks = correlation_matrix.shape[0]
    edges = []
    for i in range(number_of_stocks):
      for j in range(i + 1, number_of_stocks):
        correlation = correlation_matrix[i, j]

        distance = np.sqrt(2 * (1 - correlation))

        edges.append((distance, i, j, correlation))

    edges.sort(key=lambda edge: edge[0])

    parent = list(range(number_of_stocks))
    rank = [0] * number_of_stocks

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

    mst = np.zeros((number_of_stocks, number_of_stocks), dtype=float, )

    edges_added = 0

    for _, i, j, correlation in edges:
      if union(i, j):
        sign = np.sign(correlation)

        mst[i, j] = sign
        mst[j, i] = sign

        edges_added += 1

        if edges_added == number_of_stocks - 1:
          break

    return mst
