#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "network_builder.cpp"
#include "network_feature_extractor.cpp"
#include "ppo.cpp"
#include "trading_env.cpp"
#include "config.cpp"

using namespace std;
using namespace torch;
using namespace torch::optim;
using namespace filesystem;

Tensor node_features_to_tensor(const std::vector<std::vector<double>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = features.empty() ? 0 : static_cast<int64_t>(features[0].size());
	auto result = torch::zeros({1, stocks * feature_count, 1, stocks});
	auto accessor = result.accessor<float, 4>();
	for (int64_t stock = 0; stock < stocks; ++stock) {
		for (int64_t feature = 0; feature < feature_count; ++feature) {
			accessor[0][stock * feature_count + feature][0][stock] = static_cast<float>(features[stock][feature]);
		}
	}
	return result;
}

Tensor link_features_to_tensor(const vector<vector<vector<double>>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = (stocks == 0 || features[0].empty()) ? 0 : static_cast<int64_t>(features[0][0].size());
	auto result = torch::zeros({1, stocks * feature_count, 1, stocks, stocks});
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

Observation make_observation(TradingEnv& env, NetworkBuilder& builder) {
	vector<vector<double>> prices_network = builder.build(env.getPricesObservation());
	vector<vector<double>> volume_network = builder.build(env.getVolumeObservation());
	return {
		.x3d_price = link_features_to_tensor(NetworkFeatureExtractor::extractLinkLevelFeatures(prices_network)),
		.x3d_volume = link_features_to_tensor(NetworkFeatureExtractor::extractLinkLevelFeatures(volume_network)),
		.x2d_price = node_features_to_tensor(NetworkFeatureExtractor::extractNodeLevelFeatures(prices_network)),
		.x2d_volume = node_features_to_tensor(NetworkFeatureExtractor::extractNodeLevelFeatures(volume_network)),
		.x_portfolio = torch::tensor(env.getPercentPortfolio()).reshape({1, -1}).to(kFloat32)
	};
}

Tensor configure_allocation_weights(const Tensor& weights) {
	if (Config::WINNER_TAKES_ALL) {
		const auto best_stock = weights.argmax(-1);
		return one_hot(best_stock, weights.size(-1)).to(weights.scalar_type());
	}
	return weights;
}

void train_window(TradingEnv& env, Agent& policy, Adam& optimizer, NetworkBuilder& network_builder, const PPOConfig& ppo_config, int start_day, int end_day) {
	for (int episode = 0; episode < Config::EPISODES_PER_WINDOW; ++episode) {
		env.reset(start_day, end_day, Config::STARTING_CASH);
		RolloutBuffer rollout;
		while (!env.terminated()) {
			const Observation observation = make_observation(env, network_builder);
			const auto output = policy->act(observation);
			const auto weights = configure_allocation_weights(policy->action_to_weights(output.action)).to(kCPU);
			vector<double> action(weights.size(1));
			for (int64_t stock = 0; stock < weights.size(1); ++stock) {
				action[stock] = weights[0][stock].item<double>();
			}
			const float reward = static_cast<float>(env.step(action));
			const bool done = env.terminated() || env.portfolio_value() <= 0.0;
			rollout.add(observation, output.action, output.log_prob, output.value, reward, done);
			if (done) break;
		}

		rollout.compute_gae(ppo_config.gamma, ppo_config.lambda, 0.0f, true);
		const auto stats = ppo_update(policy, optimizer, rollout, ppo_config);
		cout << "train_window=" << start_day << ':' << end_day << " episode=" << episode + 1 << " policy_loss=" << stats.policy_loss << " value_loss=" << stats.value_loss << '\n';
	}
}

void test_window(TradingEnv& env, Agent& policy, NetworkBuilder& network_builder, int train_start_day, int start_day, int end_day, ofstream& result_csv) {
	env.reset(start_day, end_day, Config::STARTING_CASH);
	const double start_value = env.portfolio_value();
	double baseline_money = Config::STARTING_CASH;
	const size_t baseline_stock = env.getBestPerformingStock(train_start_day, start_day);
	while (!env.terminated()) {
		const Observation observation = make_observation(env, network_builder);
		const auto output = policy->act(observation, true);
		const auto next_returns = env.getNextStepReturns();
		const auto weights = configure_allocation_weights(policy->action_to_weights(output.action)).to(kCPU);
		vector<double> action(weights.size(1));
		for (int64_t stock = 0; stock < weights.size(1); ++stock) {
			action[stock] = weights[0][stock].item<double>();
		}
		env.step(action);
		baseline_money *= 1.0 + next_returns[baseline_stock];
		result_csv << env.getCurrentDate() << ',' << env.portfolio_value() << ',' << baseline_money;
		for (double weight : action) result_csv << ',' << weight * 100.0;
		result_csv << '\n';
		if (env.portfolio_value() <= 0.0) break;
	}

	cout << "test_window=" << start_day << ':' << end_day << " portfolio_start=" << start_value << " portfolio_end=" << env.portfolio_value() << '\n';
}



int main() {

	TradingEnv env(Config::TRAIN_DAYS, Config::TEST_DAYS, Config::OBS_LEN, Config::DATA_DIR, Config::STARTING_CASH);

	OutputFiles output_files(env.getStockNames());

	NetworkBuilder network_builder;
	const auto initial_observation = make_observation(env, network_builder);
	const int64_t stocks_amount = initial_observation.x_portfolio.size(1);
	const int64_t node_features_amount = initial_observation.x2d_price.size(1) / stocks_amount;
	const int64_t link_features_amount = initial_observation.x3d_price.size(1) / stocks_amount;

	PPOConfig config;
	Agent policy(stocks_amount, node_features_amount, link_features_amount);
	Adam optimizer(policy->parameters(), AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

	const int first_window_start = env.first_usable_day();
	for (int start_day = first_window_start; start_day + Config::TRAIN_DAYS + Config::TEST_DAYS <= env.total_days(); start_day += Config::TEST_DAYS) {
		const int train_end = start_day + Config::TRAIN_DAYS;
		const int test_end = train_end + Config::TEST_DAYS;
		train_window(env, policy, optimizer, network_builder, config, start_day, train_end);
		test_window(env, policy, network_builder, start_day, train_end, test_end, output_files.result_csv);
	}

	return 0;
}