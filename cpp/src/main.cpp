#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

#include "output_files.cpp"
#include "observation.cpp"
#include "baseline_calculator.cpp"

using namespace std;
using namespace torch;
using namespace torch::optim;
using namespace filesystem;

using SelectedAgent = conditional_t<Config::USE_FINANCIAL_NETWORK_MATRIX_AGENT, FinancialNetworkMatrixAgent, Agent>;

// CONSTANTS USED IN THE SHAPES AND WINDOW TIMELINE ====================================================================
// N  = Number of stocks (amount of assets in the portfolio, env.getStocksAmount())
// F  = Amount of node-level features (NetworkFeatureExtractor::nodeLevelFeaturesAmount = 8)
// L  = Amount of link-level features (NetworkFeatureExtractor::linkLevelFeaturesAmount = 8)
// O  = Observation length in days (Config::OBS_LEN = 50)
//
// FLOW ================================================================================================================
// 1. Train window: RolloutBuffer collects observations, actions, log_probs, values, and rewards over episodes.
//    GAE calculates advantages and returns. PPO update computes clipped surrogate policy loss, value loss, and entropy loss.
//    The selected agent processes either convolutional features or the financial network matrices directly.
// 2. Test window: Evaluates agent against baseline stock selection (best return over last O days) using baseline buy/sell.

int main() {

	TradingEnv env;
	OutputFiles outputFiles(env.getStockNames());
	PPOConfig config;
	SelectedAgent agent(env.getStocksAmount(), NetworkFeatureExtractor::nodeLevelFeaturesAmount, NetworkFeatureExtractor::linkLevelFeaturesAmount);
	Adam optimizer(agent->parameters(), AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

	double testingCash = Config::STARTING_CASH;
	double baselineCash = Config::STARTING_CASH;

	for (int step = env.firstUsableDay(); (step + Config::TRAIN_DAYS + Config::TEST_DAYS) <= env.getTotalDays(); step += Config::TEST_DAYS) {
		int train_start = step;
		int test_start = step + Config::TRAIN_DAYS;
		int test_end = test_start + Config::TEST_DAYS;

		cout << "Window: " << train_start << " - " << test_start << " - " << test_end << endl;

		// TRAIN ===========================================================================================================
		for (int episode = 0; episode < Config::EPISODES_PER_WINDOW; ++episode) {
			RolloutBuffer rollout;
			bool done = false;
			env.resetTraining(train_start, test_start);
			while (!done) {
				Observation observation = makeObservation(env);
				ActionOutput output = agent->act(observation);
				float reward = env.step(output.weights);
				done = env.terminated() || env.portfolio_value() <= Config::BANKRUPTCY_THRESHOLD;
				rollout.add(observation, output.action, output.log_prob, output.value, reward, done);
			}
			rollout.computeGAE(0.0f, true);
			PPOStats stats = ppoUpdate(agent, optimizer, rollout);
			cout << "\tTraining === episode=" << episode + 1 << " policy_loss=" << stats.policy_loss << " value_loss=" << stats.value_loss << '\n';
		}

		// TEST ============================================================================================================
		bool testing_done = (testingCash <= Config::BANKRUPTCY_THRESHOLD || testingCash >= Config::RESET_CASH_THRESHOLD);
		bool baseline_done = (baselineCash <= Config::BANKRUPTCY_THRESHOLD || baselineCash >= Config::RESET_CASH_THRESHOLD);
		if (testing_done && baseline_done) {
			testingCash = Config::STARTING_CASH;
			baselineCash = Config::STARTING_CASH;
		}

		env.reset(test_start, test_end, testingCash);
		double start_value = env.portfolio_value();
		double baseline_money = baselineCash;

		while (!env.terminated()) {

			Observation observation = makeObservation(env);

			vector<double> ppo_action = agent->act(observation, true);
			vector<double> baseline_same_percent = BaselineCalculator::calculateSamePercentBaseline(observation);
			vector<double> baselinebest_stock = BaselineCalculator::calculateBestStockBaseline(observation);

			size_t baseline_stock = env.getBestPerformingStock();
			baseline_money = env.baselineBuyAndSell(baseline_money, baseline_stock);

			env.step(ppo_action.weights);

			outputFiles.result_csv << env.getCurrentDate() << ',' << env.portfolio_value() << ',' << baseline_money;
			for (double weight : ppo_action.weights)
				outputFiles.result_csv << ',' << weight * 100.0;
			outputFiles.result_csv << '\n';

			bool agent_broken_or_surpassed = (env.portfolio_value() <= Config::BANKRUPTCY_THRESHOLD || env.portfolio_value() >= Config::RESET_CASH_THRESHOLD);
			bool baseline_broken_or_surpassed = (baseline_money <= Config::BANKRUPTCY_THRESHOLD || baseline_money >= Config::RESET_CASH_THRESHOLD);

			// Reset only if both break or surpass; if one is still alive continue
			if (agent_broken_or_surpassed || baseline_broken_or_surpassed) {
				env.resetMoney(Config::STARTING_CASH);
				baseline_money = Config::STARTING_CASH;
			}
		}
		testingCash = env.portfolio_value();
		baselineCash = baseline_money;
		cout << "Testing === portfolio_start=" << start_value << " portfolio_end=" << env.portfolio_value() << '\n';
	}

	return 0;
}