#include <algorithm>
#include <iostream>
#include <vector>

#include "NetworkBuilder.hpp"
#include "NetworkFeatureExtractor.hpp"
#include "PPO.cpp"
#include "TradingEnv.hpp"

namespace {

torch::Tensor node_features_to_tensor(const std::vector<std::vector<double>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count = features.empty() ? 0 : static_cast<int64_t>(features[0].size());
	auto result = torch::zeros({1, stocks * feature_count, 1, stocks});
	auto accessor = result.accessor<float, 4>();
	for (int64_t stock = 0; stock < stocks; ++stock) {
		for (int64_t feature = 0; feature < feature_count; ++feature) {
			accessor[0][stock * feature_count + feature][0][stock] =
					static_cast<float>(features[stock][feature]);
		}
	}
	return result;
}

torch::Tensor link_features_to_tensor(
		const std::vector<std::vector<std::vector<double>>>& features) {
	const int64_t stocks = static_cast<int64_t>(features.size());
	const int64_t feature_count =
			(stocks == 0 || features[0].empty()) ? 0 : static_cast<int64_t>(features[0][0].size());
	auto result = torch::zeros({1, stocks * feature_count, 1, stocks, stocks});
	auto accessor = result.accessor<float, 5>();
	for (int64_t source = 0; source < stocks; ++source) {
		for (int64_t target = 0; target < stocks; ++target) {
			for (int64_t feature = 0; feature < feature_count; ++feature) {
				accessor[0][source * feature_count + feature][0][source][target] =
						static_cast<float>(features[source][target][feature]);
			}
		}
	}
	return result;
}

Observation make_observation(TradingEnv& env, NetworkBuilder& builder) {
	const auto prices_network = builder.build(env.getPricesObservation());
	const auto volume_network = builder.build(env.getVolumeObservation());

	return {
			link_features_to_tensor(NetworkFeatureExtractor::extractLinkLevelFeatures(prices_network)),
			link_features_to_tensor(NetworkFeatureExtractor::extractLinkLevelFeatures(volume_network)),
			node_features_to_tensor(NetworkFeatureExtractor::extractNodeLevelFeatures(prices_network)),
			node_features_to_tensor(NetworkFeatureExtractor::extractNodeLevelFeatures(volume_network)),
			torch::tensor(env.getPercentPortfolio()).reshape({1, -1}).to(torch::kFloat32)};
}

}  // namespace

struct TrainingConfig {
	static constexpr int TRAIN_DAYS = 3 * 252;
	static constexpr int TEST_DAYS = 126;
	static constexpr int EPISODES_PER_WINDOW = 5;
};

void train_window(TradingEnv& env, Agent& policy, torch::optim::Adam& optimizer,
					NetworkBuilder& network_builder, const PPOConfig& ppo_config,
					int start_day, int end_day) {
	for (int episode = 0; episode < TrainingConfig::EPISODES_PER_WINDOW; ++episode) {
		env.reset(start_day, end_day);
		RolloutBuffer rollout;
		while (!env.terminated()) {
			const Observation observation = make_observation(env, network_builder);
			const auto output = policy->act(observation);
			const auto weights = policy->action_to_weights(output.action).to(torch::kCPU);
			std::vector<double> action(weights.size(1));
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
		std::cout << "train_window=" << start_day << ':' << end_day
				  << " episode=" << episode + 1
				  << " policy_loss=" << stats.policy_loss
				  << " value_loss=" << stats.value_loss << '\n';
	}
}

double test_window(TradingEnv& env, Agent& policy, NetworkBuilder& network_builder,
				   int start_day, int end_day) {
	env.reset(start_day, end_day);
	const double start_value = env.portfolio_value();
	while (!env.terminated()) {
		const Observation observation = make_observation(env, network_builder);
		const auto output = policy->act(observation, true);
		const auto weights = policy->action_to_weights(output.action).to(torch::kCPU);
		std::vector<double> action(weights.size(1));
		for (int64_t stock = 0; stock < weights.size(1); ++stock) {
			action[stock] = weights[0][stock].item<double>();
		}
		env.step(action);
		if (env.portfolio_value() <= 0.0) break;
	}

	std::cout << "test_window=" << start_day << ':' << end_day
				  << " portfolio_start=" << start_value
				  << " portfolio_end=" << env.portfolio_value() << '\n';
	return env.portfolio_value();
}

int main() {
	TradingEnv env(TrainingConfig::TRAIN_DAYS, TrainingConfig::TEST_DAYS);
	NetworkBuilder network_builder;

	const auto initial_observation = make_observation(env, network_builder);
	const int64_t stocks = initial_observation.x_portfolio.size(1);
	const int64_t node_features = initial_observation.x2d_price.size(1) / stocks;
	const int64_t link_features = initial_observation.x3d_price.size(1) / stocks;

	PPOConfig config;
	Agent policy(stocks, node_features, link_features);
	torch::optim::Adam optimizer(
			policy->parameters(), torch::optim::AdamOptions(config.learning_rate));

	const int first_window_start = env.first_usable_day();
	for (int start_day = first_window_start;
		 start_day + TrainingConfig::TRAIN_DAYS + TrainingConfig::TEST_DAYS <= env.total_days();
		 start_day += TrainingConfig::TEST_DAYS) {
		const int train_end = start_day + TrainingConfig::TRAIN_DAYS;
		const int test_end = train_end + TrainingConfig::TEST_DAYS;
		train_window(env, policy, optimizer, network_builder, config, start_day, train_end);
		test_window(env, policy, network_builder, train_end, test_end);
	}

	return 0;
}