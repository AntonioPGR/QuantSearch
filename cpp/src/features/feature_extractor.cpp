#include "feature_extractor.hpp"

#include <stdexcept>
#include <utility>

#include "network_builder.hpp"
#include "network_feature_extractor.hpp"

namespace quantsearch {

AgentInput FeatureExtractor::extract(const Matrix& prices, const Matrix& volumes) {
	if (prices.empty() || volumes.empty()) throw std::invalid_argument("Cannot extract features from empty observations");
	Matrix price_network = NetworkBuilder::build(prices);
	Matrix volume_network = NetworkBuilder::build(volumes);
	Vector price_returns(prices.front().size(), 0.0);
	Vector average_volume(volumes.front().size(), 0.0);
	for (std::size_t day = 0; day < prices.size(); ++day) {
		if (prices[day].size() != price_returns.size() || volumes[day].size() != average_volume.size()) {
			throw std::invalid_argument("Observation matrices have inconsistent stock counts");
		}
		for (std::size_t stock = 0; stock < price_returns.size(); ++stock) {
			if (day > 0) price_returns[stock] += prices[day][stock] / prices[day - 1][stock] - 1.0;
			average_volume[stock] += volumes[day][stock];
		}
	}
	if (prices.size() > 1) {
		for (double& value : price_returns) value /= static_cast<double>(prices.size() - 1);
	}
	for (double& value : average_volume) value /= static_cast<double>(volumes.size());

	return {
		{"close_prices", prices},
		{"volumes", volumes},
		{"price_network", price_network},
		{"volume_network", volume_network},
		{"price_node_features", NetworkFeatureExtractor::extractNodeLevelFeatures(price_network)},
		{"volume_node_features", NetworkFeatureExtractor::extractNodeLevelFeatures(volume_network)},
		{"price_link_features", NetworkFeatureExtractor::extractLinkLevelFeatures(price_network)},
		{"volume_link_features", NetworkFeatureExtractor::extractLinkLevelFeatures(volume_network)},
		{"average_price_returns", std::move(price_returns)},
		{"average_volume", std::move(average_volume)}
	};
}

} // namespace quantsearch
