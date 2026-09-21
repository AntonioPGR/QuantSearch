#pragma once
#ifndef QUANTSEARCH_NETWORKFEATUREEXTRACTOR_H
#define QUANTSEARCH_NETWORKFEATUREEXTRACTOR_H

#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class NetworkFeatureExtractor {
	public:

		//==================================================================================================================
		// [stock1 -> [Node Degree, Weighted Node Degree, Average Neighbor Degree, Propensity of increase degree, Node Betweenes, Node Closeness, Node Eigenvector, Node Clustering Coefficient]]
		static vector<vector<double>> extractNodeLevelFeatures(vector<vector<double> > graph) {
			int n = graph.size();
			vector<vector<double>> res(n, vector<double>(8, 0.0));
			vector<vector<int> > neighbors(n);

			// Basic Degree Calculations
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (graph[i][j] != 0) {
						neighbors[i].push_back(j);
						// 1. Node Degree
						res[i][0]++;
						// 2. Weighted Node Degree (Absolute value used for negative correlation weights)
						res[i][1] += abs(graph[i][j]);
					}
				}
			}

			// Averages and Propensity
			for (int i = 0; i < n; i++) {
				double sum_neighbor_degree = 0;
				for (int j: neighbors[i]) sum_neighbor_degree += res[j][0];
				// 3. Average Neighbor Degree
				res[i][2] = (res[i][0] > 0) ? sum_neighbor_degree / res[i][0] : 0.0;
				// 4. Propensity of increase degree
				res[i][3] = (res[i][1] != 0) ? res[i][0] / res[i][1] : 0.0;
			}

			// Shortest Paths (Floyd-Warshall)
			vector<vector<double>> dist(n, vector<double>(n, 1e9));
			vector<vector<double>> sigma(n, vector<double>(n, 0));
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i == j) {
						dist[i][j] = 0;
						sigma[i][j] = 1;
					} else if (graph[i][j] != 0) {
						dist[i][j] = 1;
						sigma[i][j] = 1;
					}
				}
			}

			for (int k = 0; k < n; k++) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						if (dist[i][k] + dist[k][j] < dist[i][j]) {
							dist[i][j] = dist[i][k] + dist[k][j];
							sigma[i][j] = sigma[i][k] * sigma[k][j];
						} else if (dist[i][k] + dist[k][j] == dist[i][j]) {
							sigma[i][j] += sigma[i][k] * sigma[k][j];
						}
					}
				}
			}

			// Node Betweenness and Closeness
			for (int v = 0; v < n; v++) {
				double betweenness = 0.0;
				double sum_dist = 0.0;
				for (int i = 0; i < n; i++) {
					if (i != v && dist[v][i] < 1e9) {
						sum_dist += dist[v][i];
					}
					for (int j = 0; j < n; j++) {
						if (i != v && j != v && dist[i][j] < 1e9 && sigma[i][j] > 0) {
							if (dist[i][v] + dist[v][j] == dist[i][j]) {
								betweenness += (sigma[i][v] * sigma[v][j]) / sigma[i][j];
							}
						}
					}
				}
				// 5. Node Betweenness
				res[v][4] = betweenness;
				// 6. Node Closeness
				res[v][5] = (sum_dist > 0) ? (n - 1) / sum_dist : 0.0; /
			}

			// 7. Node Eigenvector
			vector<double> eigen(n, 1.0 / n);
			for (int iter = 0; iter < 20; iter++) {
				vector<double> next_eigen(n, 0.0);
				double norm = 0.0;
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						if (graph[i][j] != 0) {
							next_eigen[i] += eigen[j];
						}
					}
					norm += next_eigen[i] * next_eigen[i];
				}
				norm = sqrt(norm);
				if (norm > 0) {
					for (int i = 0; i < n; i++) eigen[i] = next_eigen[i] / norm;
				}
			}
			for (int i = 0; i < n; i++) res[i][6] = eigen[i];

			// 8. Node Clustering Coefficient
			for (int i = 0; i < n; i++) {
				double possible_edges = res[i][0] * (res[i][0] - 1);
				double actual_edges = 0;
				if (possible_edges > 0) {
					for (int j: neighbors[i]) {
						for (int k: neighbors[i]) {
							if (j != k && graph[j][k] != 0) {
								actual_edges++;
							}
						}
					}
					res[i][7] = actual_edges / possible_edges;
				}
			}

			return res;
		}


		//==================================================================================================================
		// [stock1 -> [stock2 -> [Link Existence, Correlation Value, Common neighbors, Jaccard Coefficient, Adamic-Adar Coefficient, Sorenson-Dice Coefficient, Edge Betweenness, Same Community, Preferential Attachment ]]]
		static vector<vector<vector<double> > > extractLinkLevelFeatures(vector<vector<double> > graph) {
			int n = graph.size();
			vector<vector<vector<double>>> res(n, vector<vector<double> >(n, vector<double>(9, 0.0)));

			vector<int> degree(n, 0);
			vector<vector<int> > neighbors(n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (graph[i][j] != 0) {
						degree[i]++;
						neighbors[i].push_back(j);
					}
				}
			}

			// Shortest paths strictly for Edge Betweenness
			vector<vector<double> > dist(n, vector<double>(n, 1e9));
			vector<vector<double> > sigma(n, vector<double>(n, 0));
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i == j) {
						dist[i][j] = 0;
						sigma[i][j] = 1;
					} else if (graph[i][j] != 0) {
						dist[i][j] = 1;
						sigma[i][j] = 1;
					}
				}
			}
			for (int k = 0; k < n; k++) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						if (dist[i][k] + dist[k][j] < dist[i][j]) {
							dist[i][j] = dist[i][k] + dist[k][j];
							sigma[i][j] = sigma[i][k] * sigma[k][j];
						} else if (dist[i][k] + dist[k][j] == dist[i][j]) {
							sigma[i][j] += sigma[i][k] * sigma[k][j];
						}
					}
				}
			}

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i == j) continue;

					res[i][j][0] = (graph[i][j] != 0) ? 1.0 : 0.0;
					res[i][j][1] = graph[i][j];

					int common_neighbors = 0;
					double adamic_adar = 0.0;
					int ptr_i = 0, ptr_j = 0;
					while (ptr_i < neighbors[i].size() && ptr_j < neighbors[j].size()) {
						if (neighbors[i][ptr_i] == neighbors[j][ptr_j]) {
							int k = neighbors[i][ptr_i];
							common_neighbors++;
							if (degree[k] > 1) {
								adamic_adar += 1.0 / log((double) degree[k]);
							}
							ptr_i++;
							ptr_j++;
						} else if (neighbors[i][ptr_i] < neighbors[j][ptr_j]) {
							ptr_i++;
						} else {
							ptr_j++;
						}
					}

					res[i][j][2] = common_neighbors;

					int union_size = degree[i] + degree[j] - common_neighbors;
						res[i][j][3] = (union_size == 0) ? 0.0 : (double) common_neighbors / union_size;

					res[i][j][4] = adamic_adar;

					res[i][j][5] = (degree[i] + degree[j] == 0) ? 0.0 : (2.0 * common_neighbors) / (degree[i] + degree[j]);

					// Edge Betweenness
					double edge_btw = 0.0;
					if (graph[i][j] != 0) {
						for (int s = 0; s < n; s++) {
							for (int t = 0; t < n; t++) {
								if (s != t && dist[s][t] < 1e9 && sigma[s][t] > 0) {
									if (dist[s][i] + 1 + dist[j][t] == dist[s][t]) {
										edge_btw += (sigma[s][i] * sigma[j][t]) / sigma[s][t];
									}
								}
							}
						}
					}
					res[i][j][6] = edge_btw;

					// Same Community
					res[i][j][7] = 0.0; // TODO: implement louvain algorithm

					// Preferential Attachment
					res[i][j][8] = degree[i] * degree[j];
				}
			}
			return res;
		}
};

#endif //QUANTSEARCH_NETWORKFEATUREEXTRACTOR_H
