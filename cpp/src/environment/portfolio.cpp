#include "portfolio.hpp"

#include <cmath>
#include <stdexcept>

#include "../config.hpp"

namespace quantsearch {

double applyPortfolioAction(double money, const Vector& weights,
                            const Vector& current_prices, const Vector& next_prices) {
	if (weights.empty() || weights.size() != current_prices.size() || weights.size() != next_prices.size()) {
		throw std::invalid_argument("Portfolio weights and market prices must have the same nonzero size");
	}
	if (!std::isfinite(money) || money < 0.0) throw std::invalid_argument("Portfolio money must be finite and nonnegative");

	double weight_sum = 0.0;
	double gross_return = 0.0;
	for (std::size_t stock = 0; stock < weights.size(); ++stock) {
		if (!std::isfinite(weights[stock]) || weights[stock] < 0.0 ||
		    !std::isfinite(current_prices[stock]) || current_prices[stock] <= 0.0 ||
		    !std::isfinite(next_prices[stock]) || next_prices[stock] <= 0.0) {
			throw std::invalid_argument("Portfolio action or market prices contain invalid values");
		}
		weight_sum += weights[stock];
	}
	if (!std::isfinite(weight_sum) || weight_sum <= 0.0) throw std::invalid_argument("Portfolio weights must sum to a positive value");

	for (std::size_t stock = 0; stock < weights.size(); ++stock) {
		gross_return += (weights[stock] / weight_sum) * next_prices[stock] / current_prices[stock];
	}
	double next_money = money * gross_return;
	if (std::abs(next_money - money) > Config::MAX_DAILY_VALUE_CHANGE) next_money = money;
	if (!std::isfinite(next_money) || next_money <= Config::BANKRUPTCY_THRESHOLD) return 0.0;
	return next_money;
}

} // namespace quantsearch
