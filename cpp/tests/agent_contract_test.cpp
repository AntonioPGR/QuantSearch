#include <cassert>
#include <cmath>
#include <stdexcept>

#include "../src/agents/baseline_agents.hpp"
#include "../src/agents/ppo_matrix_agent.hpp"
#include "../src/config.hpp"
#include "../src/environment/portfolio.hpp"
#include "../src/features/feature_extractor.hpp"

int main() {
	using namespace quantsearch;

	const Matrix prices = {
		{10.0, 20.0, 30.0},
		{11.0, 19.0, 31.0},
		{12.0, 18.0, 33.0},
		{13.0, 17.0, 36.0}
	};
	const Matrix volumes = {
		{100.0, 200.0, 300.0},
		{110.0, 190.0, 310.0},
		{120.0, 180.0, 330.0},
		{130.0, 170.0, 360.0}
	};
	const AgentInput input = FeatureExtractor::extract(prices, volumes);
	assert(std::holds_alternative<Matrix>(input.at("price_network")));
	assert(std::holds_alternative<Vector>(input.at("average_price_returns")));
	assert(std::holds_alternative<Matrix3D>(input.at("price_link_features")));

	SamePercentAgent equal_weight(3);
	const Vector equal_weights = equal_weight.act(input);
	assert(equal_weights.size() == 3);
	assert(std::abs(equal_weights[0] - 1.0 / 3.0) < 1e-12);

	BestPerformingAgent best_stock(3);
	const Vector best_weights = best_stock.act(input);
	assert(best_weights[0] == 1.0);
	assert(best_weights[1] == 0.0);
	assert(best_weights[2] == 0.0);

	Matrix close_history(Config::OBS_LEN, Vector(3));
	Matrix volume_history(Config::OBS_LEN, Vector(3));
	for (std::size_t day = 0; day < close_history.size(); ++day) {
		for (std::size_t stock = 0; stock < 3; ++stock) {
			close_history[day][stock] = prices[day % prices.size()][stock] + static_cast<double>(day);
			volume_history[day][stock] = volumes[day % volumes.size()][stock] + static_cast<double>(day * 2);
		}
	}
	AgentInput ppo_input = input;
	ppo_input["close_prices"] = close_history;
	ppo_input["volumes"] = volume_history;
	std::vector<std::unique_ptr<Agent>> neural_agents;
	neural_agents.push_back(std::make_unique<PriceMatrixAgent>(3));
	neural_agents.push_back(std::make_unique<VolumeMatrixAgent>(3));
	neural_agents.push_back(std::make_unique<DualMatrixAgent>(3));
	neural_agents.push_back(std::make_unique<ConvolutionalAgent>(3));
	neural_agents.push_back(std::make_unique<ConvolutionalPriceAgent>(3));
	neural_agents.push_back(std::make_unique<ConvolutionalVolumeAgent>(3));
	for (auto& agent : neural_agents) {
		const Vector weights = agent->act(ppo_input);
		assert(weights.size() == 3);
		double total_weight = 0.0;
		for (double weight : weights) {
			assert(std::isfinite(weight));
			assert(weight >= 0.0);
			total_weight += weight;
		}
		assert(std::abs(total_weight - 1.0) < 1e-5);
	}

	auto* trainable = dynamic_cast<TrainableAgent*>(neural_agents[2].get());
	assert(trainable != nullptr);
	trainable->setTraining(true);
	trainable->beginEpisode();
	trainable->act(ppo_input);
	trainable->recordReward(0.0, true);
	const TrainingStats stats = trainable->updatePolicy();
	assert(std::isfinite(stats.policy_loss));
	assert(std::isfinite(stats.value_loss));
	trainable->setTraining(false);

	const double next_money = applyPortfolioAction(100.0, {0.5, 0.5}, {10.0, 20.0}, {11.0, 22.0});
	assert(std::abs(next_money - 110.0) < 1e-12);
	bool rejected_invalid_weights = false;
	try {
		applyPortfolioAction(100.0, {0.0, 0.0}, {10.0, 20.0}, {11.0, 22.0});
	} catch (const std::invalid_argument&) {
		rejected_invalid_weights = true;
	}
	assert(rejected_invalid_weights);
	return 0;
}
