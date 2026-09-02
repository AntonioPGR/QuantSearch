import pandas as pd

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy


def extract_global_network_features(adj_matrix_path):
    adj_df = pd.read_csv(adj_matrix_path, index_col=0)

    # Ensure binary unweighted graph representation
    binary_adj = adj_df.map(lambda x: 1 if x != 0 else 0)
    G = nx.from_pandas_adjacency(binary_adj)

    # 2. Extract Basic Elements
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Handle empty graphs (common in DTN during calm periods)
    if num_edges == 0:
        return {k: 0.0 for k in [
            'network_density', 'avg_clustering', 'component_count', 'giant_component_ratio',
            'avg_path_length', 'avg_degree', 'std_degree', 'avg_node_betweenness',
            'avg_node_closeness', 'avg_node_eigenvector', 'avg_edge_betweenness',
            'avg_jaccard_coef', 'avg_adamic_adar', 'avg_pref_attachment',
            'community_integration_ratio', 'degree_entropy'
        ]}

    # --- Category 1: Global Topology & Connectivity ---
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)

    # Components
    components = list(nx.connected_components(G))
    comp_count = len(components)
    giant_comp = max(components, key=len)
    giant_comp_ratio = len(giant_comp) / num_nodes

    # Path length (calculated on the Giant Component to avoid infinity on disconnected graphs)
    G_giant = G.subgraph(giant_comp)
    if len(G_giant) > 1:
        avg_path_length = nx.average_shortest_path_length(G_giant)
    else:
        avg_path_length = 0.0

    # --- Category 2: Global Node Distributions ---
    degrees = np.array([d for _, d in G.degree()])
    avg_degree = np.mean(degrees)
    std_degree = np.std(degrees)

    node_betweenness = np.array(list(nx.betweenness_centrality(G).values()))
    avg_node_betweenness = np.mean(node_betweenness)

    node_closeness = np.array(list(nx.closeness_centrality(G).values()))
    avg_node_closeness = np.mean(node_closeness)

    try:
        node_eigenvector = np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        avg_node_eigenvector = np.mean(node_eigenvector)
    except nx.PowerIterationFailedConvergence:
        avg_node_eigenvector = 0.0

    # --- Category 3: Global Link Property Aggregations ---
    # Edge Betweenness
    edge_bet_dict = nx.edge_betweenness_centrality(G)
    avg_edge_betweenness = np.mean(list(edge_bet_dict.values())) if edge_bet_dict else 0.0

    # Jaccard and Preferential Attachment (computed across all non-connected pairs)
    non_edges = list(nx.non_edges(G))
    if non_edges:
        jaccard_list = [p for _, _, p in nx.jaccard_coefficient(G, non_edges)]
        avg_jaccard = np.mean(jaccard_list)

        pref_attach_list = [p for _, _, p in nx.preferential_attachment(G, non_edges)]
        avg_pref_attachment = np.mean(pref_attach_list)

        # Adamic-Adar
        try:
            adamic_adar_list = [p for _, _, p in nx.adamic_adar_index(G, non_edges)]
            avg_adamic_adar = np.mean(adamic_adar_list)
        except ZeroDivisionError:
            avg_adamic_adar = 0.0
    else:
        avg_jaccard = 1.0
        avg_pref_attachment = avg_degree ** 2
        avg_adamic_adar = 0.0

    # Community Integration (Louvain method)
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G)
        same_community_count = 0
        total_pairs = 0
        nodes_list = list(G.nodes())

        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                total_pairs += 1
                if partition[nodes_list[i]] == partition[nodes_list[j]]:
                    same_community_count += 1
        comm_integration = same_community_count / total_pairs
    except ImportError:
        comm_integration = 0.0

    # --- Category 4: Information Entropy ---
    degree_counts = np.bincount(degrees)
    degree_probs = degree_counts / np.sum(degree_counts)
    deg_entropy = entropy(degree_probs, base=2)

    return {
        'network_density': density,
        'avg_clustering': avg_clustering,
        'component_count': comp_count,
        'giant_component_ratio': giant_comp_ratio,
        'avg_path_length': avg_path_length,
        'avg_degree': avg_degree,
        'std_degree': std_degree,
        'avg_node_betweenness': avg_node_betweenness,
        'avg_node_closeness': avg_node_closeness,
        'avg_node_eigenvector': avg_node_eigenvector,
        'avg_edge_betweenness': avg_edge_betweenness,
        'avg_jaccard_coef': avg_jaccard,
        'avg_adamic_adar': avg_adamic_adar,
        'avg_pref_attachment': avg_pref_attachment,
        'community_integration_ratio': comm_integration,
        'degree_entropy': deg_entropy
    }