#include <bits/stdc++.h>

#include "NetworkBuilder.h"
#include "NetworkFeatureExtractor.h"
#include "Printer.h"
#include "TradingEnv.h"
#include "VolumeFeatureExtractor.h"

using namespace std;

int main() {

	NetworkBuilder networkBuilder = NetworkBuilder();

	TradingEnv env = TradingEnv(0.8f);

	while (!env.train_terminated()) {

		vector<vector<double>> prices_obs = env.getPricesObservation();
		vector<vector<double>> volume_obs = env.getVolumeObservation();

		vector<vector<double>> prices_network = networkBuilder.build(prices_obs);
		vector<vector<double>> volume_network = networkBuilder.build(volume_obs);

		vector<vector<double>> prices_node_level_features = NetworkFeatureExtractor::extractNodeLevelFeatures(prices_network);
		vector<vector<vector<double>>> prices_link_level_features = NetworkFeatureExtractor::extractLinkLevelFeatures(prices_network);
		vector<vector<double>> volume_node_level_features = NetworkFeatureExtractor::extractNodeLevelFeatures(volume_network);
		vector<vector<vector<double>>> volume_link_level_features = NetworkFeatureExtractor::extractLinkLevelFeatures(volume_network);

		vector<double> portfolio = env.getPercentPortfolio();
		// vector<double> convolutional_price_features = convolutionalFeatureExtractor.extract(prices_obs);
		// vector<double> convolutional_volume_features = convolutionalFeatureExtractor.extract(volume_obs);
		// vector<double> network_price_features = networkFeatureExtractor.extract(prices_network);
		// vector<double> network_volume_features = networkFeatureExtractor.extract(volume_network);
		// vector<double> price_features = priceFeatureExtractor.extract(prices_obs);
		// vector<double> volume_features = volumeFeatureExtractor.extract(prices_obs);
		break;

	}

	return 0;
}