#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "network_builder.cpp"
#include "network_feature_extractor.cpp"
#include "ppo.cpp"
#include "trading_env.cpp"

using namespace std;
using namespace torch;
using namespace torch::optim;

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

struct TrainingConfig {
	static constexpr int TRAIN_DAYS = 3 * 252;
	static constexpr int TEST_DAYS = 126;
	static constexpr int EPISODES_PER_WINDOW = 8;
};

struct TestResult {
	double policy_money;
	double baseline_money;
};

void train_window(TradingEnv& env, Agent& policy, optim::Adam& optimizer, NetworkBuilder& network_builder, const PPOConfig& ppo_config, int start_day, int end_day, ofstream& training_log) {
	for (int episode = 0; episode < TrainingConfig::EPISODES_PER_WINDOW; ++episode) {
		env.reset(start_day, end_day);
		RolloutBuffer rollout;
		while (!env.terminated()) {
			const Observation observation = make_observation(env, network_builder);
			const auto output = policy->act(observation);
			const auto weights = policy->action_to_weights(output.action).to(kCPU);
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
		training_log << "train," << start_day << ',' << end_day << ',' << episode + 1 << ','
		             << stats.policy_loss << ',' << stats.value_loss << ',' << stats.entropy << ','
		             << stats.approx_kl << ",,,,\n";
		cout << "train_window=" << start_day << ':' << end_day << " episode=" << episode + 1 << " policy_loss=" << stats.policy_loss << " value_loss=" << stats.value_loss << '\n';
	}
}

TestResult test_window(TradingEnv& env, Agent& policy, NetworkBuilder& network_builder, int start_day, int end_day, double starting_money, double baseline_starting_money, ofstream& result_csv, ofstream& training_log) {
	env.reset(start_day, end_day, starting_money);
	const double start_value = env.portfolio_value();
	double baseline_money = baseline_starting_money;
	const double baseline_start_value = baseline_money;
	while (!env.terminated()) {
		const Observation observation = make_observation(env, network_builder);
		const auto output = policy->act(observation, true);
		const auto next_returns = env.getNextStepReturns();
		const auto best_stock = static_cast<size_t>(max_element(next_returns.begin(), next_returns.end()) - next_returns.begin());
		const auto weights = policy->action_to_weights(output.action).to(kCPU);
		vector<double> action(weights.size(1));
		for (int64_t stock = 0; stock < weights.size(1); ++stock) {
			action[stock] = weights[0][stock].item<double>();
		}
		env.step(action);
		baseline_money *= 1.0 + next_returns[best_stock];
		result_csv << env.getCurrentDate() << ',' << env.portfolio_value() << ',' << baseline_money;
		for (double weight : action) result_csv << ',' << weight * 100.0;
		result_csv << '\n';
		if (env.portfolio_value() <= 0.0) break;
	}

	cout << "test_window=" << start_day << ':' << end_day << " portfolio_start=" << start_value << " portfolio_end=" << env.portfolio_value() << '\n';
	training_log << "test," << start_day << ',' << end_day << ",0,,,,," << start_value << ',' << env.portfolio_value() << ',' << baseline_start_value << ',' << baseline_money << '\n';
	return {env.portfolio_value(), baseline_money};
}

int main() {
	TradingEnv env(TrainingConfig::TRAIN_DAYS, TrainingConfig::TEST_DAYS);
	filesystem::path workspace_root = filesystem::current_path();
	if (!filesystem::is_directory(workspace_root / "_data") && filesystem::is_directory(workspace_root / "../_data")) {
		workspace_root = filesystem::absolute(workspace_root / "..");
	}
	filesystem::create_directories(workspace_root / "_logs");
	filesystem::create_directories(workspace_root / "_results");
	ofstream training_log(workspace_root / "_logs" / "training_log.csv");
	ofstream result_csv(workspace_root / "_results" / "test_results.csv");
	if (!training_log || !result_csv) throw runtime_error("Could not create training or test result CSV");
	training_log << "type,train_start,train_end,episode,policy_loss,value_loss,entropy,approx_kl,portfolio_start,portfolio_end,baseline_start,baseline_end\n";
	result_csv << "date,my_money,baseline_money";
	for (const string& stock_name : env.getStockNames()) {
		result_csv << ",percent_stock_" << stock_name;
	}
	result_csv << '\n' << setprecision(12);
	training_log << setprecision(12);

	NetworkBuilder network_builder;

	const auto initial_observation = make_observation(env, network_builder);
	const int64_t stocks_amount = initial_observation.x_portfolio.size(1);
	const int64_t node_features_amount = initial_observation.x2d_price.size(1) / stocks_amount;
	const int64_t link_features_amount = initial_observation.x3d_price.size(1) / stocks_amount;

	PPOConfig config;
	Agent policy(stocks_amount, node_features_amount, link_features_amount);
	Adam optimizer(policy->parameters(), AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

	const int first_window_start = env.first_usable_day();
	double test_money = 10000.0;
	double baseline_money = 10000.0;
	for (int start_day = first_window_start; start_day + TrainingConfig::TRAIN_DAYS + TrainingConfig::TEST_DAYS <= env.total_days(); start_day += TrainingConfig::TEST_DAYS) {
		const int train_end = start_day + TrainingConfig::TRAIN_DAYS;
		const int test_end = train_end + TrainingConfig::TEST_DAYS;
		train_window(env, policy, optimizer, network_builder, config, start_day, train_end, training_log);
		const TestResult result = test_window(env, policy, network_builder, train_end, test_end, test_money, baseline_money, result_csv, training_log);
		test_money = result.policy_money;
		baseline_money = result.baseline_money;
	}

	return 0;
}