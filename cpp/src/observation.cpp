#pragma once

#include <torch/torch.h>
#include <vector>

#include "network_builder.cpp"
#include "network_feature_extractor.cpp"
#include "trading_env.cpp"

using namespace std;
using namespace torch;

// CONSTANTS USED IN THE SHAPES BELOW ==================================================================================
// B = Batch size (1 for a single observation, T when the rollout is stacked, minibatch size inside PPO)
// N = Number of stocks (amount of assets in the portfolio)
// F = Amount of node-level features (NetworkFeatureExtractor::nodeLevelFeaturesAmount)
// L = Amount of link-level features (NetworkFeatureExtractor::linkLevelFeaturesAmount)

struct Observation {
	Tensor x1d_price; // [B, F, N]    -> x1d_price[b][f] = column f of the node features matrix (feature f of every stock)
	Tensor x1d_volume;
	Tensor x2d_price; // [B, L, N, N] -> x2d_price[b][l] = matrix of link feature l between every pair of stocks
	Tensor x2d_volume;
	Tensor financial_network_price; // [B, 1, N, N] -> thresholded price-correlation network
	Tensor financial_network_volume; // [B, 1, N, N] -> thresholded volume-correlation network
	Tensor x_portfolio; // [B, N]

	Observation to(Device d) {
		return {
			.x1d_price = x1d_price.to(d),
			.x1d_volume = x1d_volume.to(d),
			.x2d_price = x2d_price.to(d),
			.x2d_volume = x2d_volume.to(d),
			.financial_network_price = financial_network_price.to(d),
			.financial_network_volume = financial_network_volume.to(d),
			.x_portfolio = x_portfolio.to(d)
		};
	}

	Observation index(Tensor& idx) {
		return {
			.x1d_price = x1d_price.index_select(0, idx),
			.x1d_volume = x1d_volume.index_select(0, idx),
			.x2d_price = x2d_price.index_select(0, idx),
			.x2d_volume = x2d_volume.index_select(0, idx),
			.financial_network_price = financial_network_price.index_select(0, idx),
			.financial_network_volume = financial_network_volume.index_select(0, idx),
			.x_portfolio = x_portfolio.index_select(0, idx)
		};
	}

	static Observation stack(const vector<Observation> &v) {
		vector<Tensor> a, b, c, d, e, f, g;
		a.reserve(v.size()); b.reserve(v.size()); c.reserve(v.size());
		d.reserve(v.size()); e.reserve(v.size()); f.reserve(v.size()); g.reserve(v.size());
		for (const Observation& o : v) {
			a.push_back(o.x1d_price);  b.push_back(o.x1d_volume);
			c.push_back(o.x2d_price);  d.push_back(o.x2d_volume);
			f.push_back(o.financial_network_price); g.push_back(o.financial_network_volume);
			e.push_back(o.x_portfolio);
		}
		return {
			.x1d_price = cat(a, 0),
			.x1d_volume = cat(b, 0),
			.x2d_price = cat(c, 0),
			.x2d_volume = cat(d, 0),
			.financial_network_price = cat(f, 0),
			.financial_network_volume = cat(g, 0),
			.x_portfolio = cat(e, 0)
		};
	}
};

// features[N][F] (row = stock, column = node feature) -> [1, F, N] (every channel f is the column f, one value per stock)
inline Tensor features2dToTensor(const vector<vector<double>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = static_cast<int64_t>(features[0].size());
	auto result = torch::zeros({1, feature_count, stocks});
	auto accessor = result.accessor<float, 3>();
	for (int64_t feature = 0; feature < feature_count; ++feature) {
		for (int64_t stock = 0; stock < stocks; ++stock) {
			accessor[0][feature][stock] = static_cast<float>(features[stock][feature]);
		}
	}
	return result;
}

// features[N][N][L] (source stock, target stock, link feature) -> [1, L, N, N] (every channel l is the N x N matrix of link feature l)
inline Tensor features3dToTensor(const vector<vector<vector<double>>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = (stocks == 0 || features[0].empty()) ? 0 : static_cast<int64_t>(features[0][0].size());
	auto result = torch::zeros({1, feature_count, stocks, stocks});
	auto accessor = result.accessor<float, 4>();
	for (int64_t feature = 0; feature < feature_count; ++feature) {
		for (int64_t source = 0; source < stocks; ++source) {
			for (int64_t target = 0; target < stocks; ++target) {
				accessor[0][feature][source][target] = static_cast<float>(features[source][target][feature]);
			}
		}
	}
	return result;
}

// matrix[N][N] -> [1, 1, N, N]
inline Tensor networkMatrixToTensor(const vector<vector<double>>& matrix) {
	const int64_t stocks = static_cast<int64_t>(matrix.size());
	auto result = torch::zeros({1, 1, stocks, stocks});
	auto accessor = result.accessor<float, 4>();
	for (int64_t source = 0; source < stocks; ++source) {
		for (int64_t target = 0; target < stocks; ++target) {
			accessor[0][0][source][target] = static_cast<float>(matrix[source][target]);
		}
	}
	return result;
}

inline Observation makeObservation(TradingEnv& env) {
	vector<vector<double>> prices_network = NetworkBuilder::build(env.getPricesObservation());  // [N][N]
	vector<vector<double>> volume_network = NetworkBuilder::build(env.getVolumeObservation());  // [N][N]
	const int64_t stocks = env.getStocksAmount();
	if constexpr (Config::USE_FINANCIAL_NETWORK_MATRIX_AGENT) {
		return {
			.x1d_price = torch::zeros({1, 0, stocks}),
			.x1d_volume = torch::zeros({1, 0, stocks}),
			.x2d_price = torch::zeros({1, 0, stocks, stocks}),
			.x2d_volume = torch::zeros({1, 0, stocks, stocks}),
			.financial_network_price = networkMatrixToTensor(prices_network),
			.financial_network_volume = networkMatrixToTensor(volume_network),
			.x_portfolio = torch::tensor(env.getPercentPortfolio()).reshape({1, -1}).to(kFloat32)
		};
	}
	return {
		.x1d_price = features2dToTensor(NetworkFeatureExtractor::extractNodeLevelFeatures(prices_network)),   // [1, F, N]
		.x1d_volume = features2dToTensor(NetworkFeatureExtractor::extractNodeLevelFeatures(volume_network)),  // [1, F, N]
		.x2d_price = features3dToTensor(NetworkFeatureExtractor::extractLinkLevelFeatures(prices_network)),   // [1, L, N, N]
		.x2d_volume = features3dToTensor(NetworkFeatureExtractor::extractLinkLevelFeatures(volume_network)),  // [1, L, N, N]
		.financial_network_price = networkMatrixToTensor(prices_network),
		.financial_network_volume = networkMatrixToTensor(volume_network),
		.x_portfolio = torch::tensor(env.getPercentPortfolio()).reshape({1, -1}).to(kFloat32)                 // [1, N]
	};
}
