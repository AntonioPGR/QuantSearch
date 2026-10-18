
struct Observation {
	Tensor x3d_price; // [B, n_stocks*n_link_features, D, H, W]
	Tensor x3d_volume;
	Tensor x2d_price; // [B, n_stocks*n_node_features, H, W]
	Tensor x2d_volume;
	Tensor x_portfolio; // [B, n_stocks]

	Observation to(Device d) {
		return {
			.x3d_price = x3d_price.to(d),
			.x3d_volume = x3d_volume.to(d),
			.x2d_price = x2d_price.to(d),
			.x2d_volume = x2d_volume.to(d),
			.x_portfolio = x_portfolio.to(d)
		};
	}

	Observation index(Tensor& idx) {
		return {
			.x3d_price = x3d_price.index_select(0, idx),
			.x3d_volume = x3d_volume.index_select(0, idx),
			.x2d_price = x2d_price.index_select(0, idx),
			.x2d_volume = x2d_volume.index_select(0, idx),
			.x_portfolio = x_portfolio.index_select(0, idx)
		};
	}

	static Observation stack(const vector<Observation> &v) {
		vector<Tensor> a, b, c, d, e;
		a.reserve(v.size()); b.reserve(v.size()); c.reserve(v.size());
		d.reserve(v.size()); e.reserve(v.size());
		for (const Observation& o : v) {
			a.push_back(o.x3d_price);  b.push_back(o.x3d_volume);
			c.push_back(o.x2d_price);  d.push_back(o.x2d_volume);
			e.push_back(o.x_portfolio);
		}
		return {
			.x3d_price = cat(a, 0),
			.x3d_volume = cat(b, 0),
			.x2d_price = cat(c, 0),
			.x2d_volume = cat(d, 0),
			.x_portfolio = cat(e, 0)
		};
	}
};

Tensor features2dToTensor(const vector<vector<double>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = static_cast<int64_t>(features[0].size());
	auto result = torch::zeros({1, stocks * feature_count, 1, stocks});  // TODO -> CHANGE FORMAT TO [stocks, feature_count, 1, stocks]
	auto accessor = result.accessor<float, 4>();
	for (int64_t stock = 0; stock < stocks; ++stock) {
		for (int64_t feature = 0; feature < feature_count; ++feature) {
			accessor[0][stock * feature_count + feature][0][stock] = static_cast<float>(features[stock][feature]);
		}
	}
	return result;
}

Tensor features3dToTensor(const vector<vector<vector<double>>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = (stocks == 0 || features[0].empty()) ? 0 : static_cast<int64_t>(features[0][0].size());
	auto result = torch::zeros({1, stocks * feature_count, 1, stocks, stocks}); // TODO -> CHANGE FORMAT TO [stocks, feature_count, 1, stocks, stocks]
	auto accessor = result.accessor<float, 5>();
	for (int64_t source = 0; source < stocks; ++source) {
		for (int64_t target = 0; target < stocks; ++target) {
			for (int64_t feature = 0; feature < feature_count; ++feature) {
				accessor[0][source * feature_count + feature][0][source][target] = static_cast<float>(features[source][target][feature]);
			}
		}
	}
	return result;
}

Observation makeObservation(TradingEnv& env) {
	vector<vector<double>> prices_network = NetworkBuilder::build(env.getPricesObservation());
	vector<vector<double>> volume_network = NetworkBuilder::build(env.getVolumeObservation());
	return {
		.x3d_price = features3dToTensor(NetworkFeatureExtractor::extractLinkLevelFeatures(prices_network)),
		.x3d_volume = features3dToTensor(NetworkFeatureExtractor::extractLinkLevelFeatures(volume_network)),
		.x2d_price = features2dToTensor(NetworkFeatureExtractor::extractNodeLevelFeatures(prices_network)),
		.x2d_volume = features2dToTensor(NetworkFeatureExtractor::extractNodeLevelFeatures(volume_network)),
		.x_portfolio = torch::tensor(env.getPercentPortfolio()).reshape({1, -1}).to(kFloat32)
	};
}
