#include "network_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../config.hpp"

namespace quantsearch {

Matrix NetworkBuilder::calculateCorrelationMatrix(const Matrix& observations) {
	if (observations.empty() || observations.front().empty()) return {};
	const std::size_t stocks = observations.front().size();
	for (const auto& row : observations) {
		if (row.size() != stocks) throw std::invalid_argument("Observation matrix rows must have equal lengths");
	}

	Vector means(stocks, 0.0);
	for (const auto& row : observations) {
		for (std::size_t stock = 0; stock < stocks; ++stock) means[stock] += row[stock];
	}
	for (double& mean : means) mean /= static_cast<double>(observations.size());

	Matrix correlations(stocks, Vector(stocks, 0.0));
	for (std::size_t i = 0; i < stocks; ++i) {
		for (std::size_t j = i; j < stocks; ++j) {
			double covariance = 0.0;
			double variance_i = 0.0;
			double variance_j = 0.0;
			for (const auto& row : observations) {
				const double diff_i = row[i] - means[i];
				const double diff_j = row[j] - means[j];
				covariance += diff_i * diff_j;
				variance_i += diff_i * diff_i;
				variance_j += diff_j * diff_j;
			}
			const double denominator = std::sqrt(variance_i * variance_j);
			const double correlation = denominator == 0.0 ? 0.0 : covariance / denominator;
			correlations[i][j] = correlation;
			correlations[j][i] = correlation;
		}
	}
	return correlations;
}

Matrix NetworkBuilder::build(const Matrix& observations) {
	Matrix correlations = calculateCorrelationMatrix(observations);
	const std::size_t stocks = correlations.size();
	if (stocks < 2) return correlations;

	Vector absolute_correlations;
	for (std::size_t i = 0; i < stocks; ++i) {
		for (std::size_t j = i + 1; j < stocks; ++j) absolute_correlations.push_back(std::abs(correlations[i][j]));
	}
	const std::size_t number_to_keep = std::max<std::size_t>(1, absolute_correlations.size() / 2);
	std::sort(absolute_correlations.begin(), absolute_correlations.end());
	const double percentile_threshold = absolute_correlations[absolute_correlations.size() - number_to_keep];

	for (std::size_t i = 0; i < stocks; ++i) {
		for (std::size_t j = 0; j < stocks; ++j) {
			if (std::abs(correlations[i][j]) < percentile_threshold ||
			    std::abs(correlations[i][j]) < Config::NETWORK_THRESHOLD) correlations[i][j] = 0.0;
		}
	}
	return correlations;
}

} // namespace quantsearch
