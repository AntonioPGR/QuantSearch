#include "network_feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace quantsearch {

namespace {

struct Paths {
	Matrix distance;
	Matrix count;
};

Paths shortestPaths(const Matrix& graph) {
	const std::size_t n = graph.size();
	const double infinity = std::numeric_limits<double>::infinity();
	Paths paths{Matrix(n, Vector(n, infinity)), Matrix(n, Vector(n, 0.0))};
	for (std::size_t i = 0; i < n; ++i) {
		paths.distance[i][i] = 0.0;
		paths.count[i][i] = 1.0;
		for (std::size_t j = 0; j < n; ++j) {
			if (i != j && graph[i][j] != 0.0) {
				paths.distance[i][j] = 1.0;
				paths.count[i][j] = 1.0;
			}
		}
	}
	for (std::size_t k = 0; k < n; ++k) {
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < n; ++j) {
				const double through = paths.distance[i][k] + paths.distance[k][j];
				if (through < paths.distance[i][j]) {
					paths.distance[i][j] = through;
					paths.count[i][j] = paths.count[i][k] * paths.count[k][j];
				} else if (i != j && through == paths.distance[i][j] && std::isfinite(through)) {
					paths.count[i][j] += paths.count[i][k] * paths.count[k][j];
				}
			}
		}
	}
	return paths;
}

std::vector<std::vector<std::size_t>> neighborsOf(const Matrix& graph) {
	std::vector<std::vector<std::size_t>> neighbors(graph.size());
	for (std::size_t i = 0; i < graph.size(); ++i) {
		for (std::size_t j = 0; j < graph.size(); ++j) {
			if (i != j && graph[i][j] != 0.0) neighbors[i].push_back(j);
		}
	}
	return neighbors;
}

void validateGraph(const Matrix& graph) {
	for (const auto& row : graph) {
		if (row.size() != graph.size()) throw std::invalid_argument("Network graph must be square");
	}
}

} // namespace

Matrix NetworkFeatureExtractor::extractNodeLevelFeatures(const Matrix& graph) {
	validateGraph(graph);
	const std::size_t n = graph.size();
	Matrix features(n, Vector(nodeFeatureCount, 0.0));
	const auto neighbors = neighborsOf(graph);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j : neighbors[i]) {
			features[i][0] += 1.0;
			features[i][1] += std::abs(graph[i][j]);
		}
	}
	for (std::size_t i = 0; i < n; ++i) {
		double neighbor_degree = 0.0;
		for (std::size_t j : neighbors[i]) neighbor_degree += features[j][0];
		if (features[i][0] > 0.0) features[i][2] = neighbor_degree / features[i][0];
		if (features[i][1] > 0.0) features[i][3] = features[i][0] / features[i][1];
	}

	const Paths paths = shortestPaths(graph);
	for (std::size_t v = 0; v < n; ++v) {
		double betweenness = 0.0;
		double distance_sum = 0.0;
		for (std::size_t i = 0; i < n; ++i) {
			if (i != v && std::isfinite(paths.distance[v][i])) distance_sum += paths.distance[v][i];
			for (std::size_t j = 0; j < n; ++j) {
				if (i == v || j == v || !std::isfinite(paths.distance[i][j]) || paths.count[i][j] == 0.0) continue;
				if (paths.distance[i][v] + paths.distance[v][j] == paths.distance[i][j]) {
					betweenness += paths.count[i][v] * paths.count[v][j] / paths.count[i][j];
				}
			}
		}
		features[v][4] = betweenness;
		features[v][5] = distance_sum > 0.0 ? static_cast<double>(n - 1) / distance_sum : 0.0;
	}

	if (n > 0) {
		Vector eigenvector(n, 1.0 / static_cast<double>(n));
		for (int iteration = 0; iteration < 20; ++iteration) {
			Vector next(n, 0.0);
			double norm = 0.0;
			for (std::size_t i = 0; i < n; ++i) {
				for (std::size_t j : neighbors[i]) next[i] += eigenvector[j];
				norm += next[i] * next[i];
			}
			norm = std::sqrt(norm);
			if (norm > 0.0) for (double& value : next) value /= norm;
			eigenvector = std::move(next);
		}
		for (std::size_t i = 0; i < n; ++i) features[i][6] = eigenvector[i];
	}

	for (std::size_t i = 0; i < n; ++i) {
		const double possible = features[i][0] * (features[i][0] - 1.0);
		double actual = 0.0;
		for (std::size_t j : neighbors[i]) {
			for (std::size_t k : neighbors[i]) {
				if (j != k && graph[j][k] != 0.0) actual += 1.0;
			}
		}
		if (possible > 0.0) features[i][7] = actual / possible;
	}
	return features;
}

Matrix3D NetworkFeatureExtractor::extractLinkLevelFeatures(const Matrix& graph) {
	validateGraph(graph);
	const std::size_t n = graph.size();
	Matrix3D features(n, Matrix(n, Vector(linkFeatureCount, 0.0)));
	const auto neighbors = neighborsOf(graph);
	Vector degree(n, 0.0);
	for (std::size_t i = 0; i < n; ++i) degree[i] = static_cast<double>(neighbors[i].size());
	const Paths paths = shortestPaths(graph);

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			if (i == j) continue;
			features[i][j][0] = graph[i][j] == 0.0 ? 0.0 : 1.0;
			features[i][j][1] = graph[i][j];
			double common = 0.0;
			double adamic_adar = 0.0;
			for (std::size_t k : neighbors[i]) {
				if (std::find(neighbors[j].begin(), neighbors[j].end(), k) != neighbors[j].end()) {
					common += 1.0;
					if (degree[k] > 1.0) adamic_adar += 1.0 / std::log(degree[k]);
				}
			}
			const double union_size = degree[i] + degree[j] - common;
			const double degree_sum = degree[i] + degree[j];
			features[i][j][2] = common;
			features[i][j][3] = union_size > 0.0 ? common / union_size : 0.0;
			features[i][j][4] = adamic_adar;
			features[i][j][5] = degree_sum > 0.0 ? 2.0 * common / degree_sum : 0.0;
			features[i][j][7] = degree[i] * degree[j];

			if (graph[i][j] != 0.0) {
				double edge_betweenness = 0.0;
				for (std::size_t source = 0; source < n; ++source) {
					for (std::size_t target = 0; target < n; ++target) {
						if (source == target || !std::isfinite(paths.distance[source][target]) ||
						    paths.count[source][target] == 0.0) continue;
						if (paths.distance[source][i] + 1.0 + paths.distance[j][target] == paths.distance[source][target]) {
							edge_betweenness += paths.count[source][i] * paths.count[j][target] /
							                    paths.count[source][target];
						}
					}
				}
				features[i][j][6] = edge_betweenness;
			}
		}
	}
	return features;
}

} // namespace quantsearch
