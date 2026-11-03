#include "baseline_agents.hpp"

#include <cmath>
#include <stdexcept>

namespace quantsearch {

namespace {

const Matrix& requireMatrix(const AgentInput& input, const std::string& feature_name) {
	const auto feature = input.find(feature_name);
	if (feature == input.end() || !std::holds_alternative<Matrix>(feature->second)) {
		throw std::invalid_argument("Agent requires matrix feature '" + feature_name + "'");
	}
	return std::get<Matrix>(feature->second);
}

} // namespace

SamePercentAgent::SamePercentAgent(std::size_t stock_count) : stock_count_(stock_count) {
	if (stock_count_ == 0) throw std::invalid_argument("SamePercentAgent needs at least one stock");
}

const std::string& SamePercentAgent::name() const {
	return name_;
}

Vector SamePercentAgent::act(const AgentInput&) {
	return Vector(stock_count_, 1.0 / static_cast<double>(stock_count_));
}

BestPerformingAgent::BestPerformingAgent(std::size_t stock_count) : stock_count_(stock_count) {
	if (stock_count_ == 0) throw std::invalid_argument("BestPerformingAgent needs at least one stock");
}

const std::string& BestPerformingAgent::name() const {
	return name_;
}

Vector BestPerformingAgent::act(const AgentInput& input) {
	const Matrix& prices = requireMatrix(input, "close_prices");
	if (prices.empty() || prices.front().size() != stock_count_) {
		throw std::invalid_argument("BestPerformingAgent received a price matrix with an invalid shape");
	}

	std::size_t best_stock = 0;
	double best_return = -INFINITY;
	for (std::size_t stock = 0; stock < stock_count_; ++stock) {
		double cumulative_log_return = 0.0;
		for (std::size_t day = 1; day < prices.size(); ++day) {
			if (prices[day].size() != stock_count_ || !std::isfinite(prices[day][stock]) ||
			    !std::isfinite(prices[day - 1][stock]) || prices[day][stock] <= 0.0 ||
			    prices[day - 1][stock] <= 0.0) {
				throw std::invalid_argument("BestPerformingAgent received invalid close prices");
			}
			cumulative_log_return += std::log(prices[day][stock] / prices[day - 1][stock]);
		}
		if (cumulative_log_return > best_return) {
			best_return = cumulative_log_return;
			best_stock = stock;
		}
	}

	Vector weights(stock_count_, 0.0);
	weights[best_stock] = 1.0;
	return weights;
}

} // namespace quantsearch
