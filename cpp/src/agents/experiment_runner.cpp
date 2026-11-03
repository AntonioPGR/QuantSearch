#include "experiment_runner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../config.hpp"
#include "../environment/portfolio.hpp"
#include "../environment/trading_env.hpp"
#include "../features/feature_extractor.hpp"
#include "../output/output_files.hpp"
#include "agent.hpp"

namespace quantsearch {

namespace {

double rewardFor(double previous_money, double next_money) {
	if (next_money <= Config::BANKRUPTCY_THRESHOLD) return -1.0;
	if (previous_money <= 0.0) return 0.0;
	return std::log2(next_money / previous_money);
}

Vector validatedWeights(Agent& agent, const AgentInput& input, std::size_t stocks) {
	Vector weights = agent.act(input);
	if (weights.size() != stocks) {
		throw std::runtime_error("Agent '" + agent.name() + "' returned " + std::to_string(weights.size()) +
		                         " weights for " + std::to_string(stocks) + " stocks");
	}
	return weights;
}

} // namespace

void runExperiments() {
	TradingEnv env;
	std::vector<std::unique_ptr<Agent>> agents;
	std::vector<std::string> agent_names;
	for (std::string_view configured_name : Config::ENABLED_AGENTS) {
		auto agent = createAgent(std::string(configured_name), env.stockCount());
		if (std::find(agent_names.begin(), agent_names.end(), agent->name()) != agent_names.end()) {
			throw std::invalid_argument("Configured agent names must be unique: " + agent->name());
		}
		agent_names.push_back(agent->name());
		agents.push_back(std::move(agent));
	}
	if (env.firstUsableDay() + Config::TRAIN_DAYS + Config::TEST_DAYS > env.totalDays()) {
		throw std::runtime_error("Market history is too short for the configured training and test windows");
	}
	OutputFiles output(env.stockNames(), agent_names);
	std::vector<double> testing_money(agents.size(), Config::STARTING_CASH);
	std::vector<std::unique_ptr<AgentInput>> feature_cache(static_cast<std::size_t>(env.totalDays()));
	const auto input_for_current_day = [&]() -> const AgentInput& {
		auto& cached = feature_cache.at(static_cast<std::size_t>(env.currentStep()));
		if (!cached) {
			cached = std::make_unique<AgentInput>(
				FeatureExtractor::extract(env.getPriceObservation(), env.getVolumeObservation()));
		}
		return *cached;
	};

	for (int window_start = env.firstUsableDay();
	     window_start + Config::TRAIN_DAYS + Config::TEST_DAYS <= env.totalDays();
	     window_start += Config::TEST_DAYS) {
		const int training_end = window_start + Config::TRAIN_DAYS;
		const int test_end = training_end + Config::TEST_DAYS;
		std::cout << "Window: " << window_start << " - " << training_end << " - " << test_end << '\n';

		for (auto& agent_ptr : agents) {
			auto* trainable = dynamic_cast<TrainableAgent*>(agent_ptr.get());
			if (trainable == nullptr) continue;
			trainable->setTraining(true);
			for (int episode = 0; episode < Config::EPISODES_PER_WINDOW; ++episode) {
				env.reset(window_start, training_end);
				trainable->beginEpisode();
				double money = Config::STARTING_CASH;
				while (!env.terminated()) {
					const AgentInput& input = input_for_current_day();
					const Vector weights = validatedWeights(*agent_ptr, input, env.stockCount());
					const double next_money = applyPortfolioAction(money, weights, env.currentPrices(), env.nextPrices());
					env.advance();
					const bool terminal = env.terminated() || next_money <= Config::BANKRUPTCY_THRESHOLD;
					trainable->recordReward(rewardFor(money, next_money), terminal);
					money = next_money;
				}
				const TrainingStats stats = trainable->updatePolicy();
				std::cout << "\tTraining agent=" << agent_ptr->name() << " episode=" << episode + 1
				          << " policy_loss=" << stats.policy_loss << " value_loss=" << stats.value_loss << '\n';
			}
			trainable->setTraining(false);
		}

		env.reset(training_end, test_end);
		std::vector<AgentResult> day_results(agents.size());
		while (!env.terminated()) {
			const AgentInput& input = input_for_current_day();
			for (std::size_t index = 0; index < agents.size(); ++index) {
				day_results[index].weights = validatedWeights(*agents[index], input, env.stockCount());
				day_results[index].money = applyPortfolioAction(testing_money[index], day_results[index].weights,
				                                               env.currentPrices(), env.nextPrices());
			}
			env.advance();
			output.writeResults(env.currentDate(), day_results);
			for (std::size_t index = 0; index < agents.size(); ++index) {
				testing_money[index] = day_results[index].money;
				if (testing_money[index] <= Config::BANKRUPTCY_THRESHOLD ||
				    testing_money[index] >= Config::RESET_CASH_THRESHOLD) {
					testing_money[index] = Config::STARTING_CASH;
				}
			}
		}
		for (std::size_t index = 0; index < agents.size(); ++index) {
			std::cout << "Testing === agent=" << agents[index]->name()
			          << " portfolio_end=" << testing_money[index] << '\n';
		}
	}
}

} // namespace quantsearch
