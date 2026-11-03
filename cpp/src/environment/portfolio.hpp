#pragma once

#include "../types.hpp"

namespace quantsearch {

double applyPortfolioAction(double money, const Vector& weights,
                            const Vector& current_prices, const Vector& next_prices);

} // namespace quantsearch
