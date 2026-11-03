#include "agent.hpp"

#include <stdexcept>

#include "baseline_agents.hpp"
#include "ppo_matrix_agent.hpp"

namespace quantsearch {

std::unique_ptr<Agent> createAgent(const std::string& name, std::size_t stock_count) {
	if (name == "same_percent") return std::make_unique<SamePercentAgent>(stock_count);
	if (name == "best_performing") return std::make_unique<BestPerformingAgent>(stock_count);
	if (name == "price_matrix") return std::make_unique<PriceMatrixAgent>(stock_count);
	if (name == "volume_matrix") return std::make_unique<VolumeMatrixAgent>(stock_count);
	if (name == "dual_matrix") return std::make_unique<DualMatrixAgent>(stock_count);
	if (name == "convolutional" || name == "convolutional_both" || name == "convolutional_dual") return std::make_unique<ConvolutionalAgent>(stock_count);
	if (name == "convolutional_price" || name == "convolutional_close" || name == "convolutional_closes") return std::make_unique<ConvolutionalPriceAgent>(stock_count);
	if (name == "convolutional_volume" || name == "convolutional_volumes") return std::make_unique<ConvolutionalVolumeAgent>(stock_count);
	throw std::invalid_argument("Unknown configured agent: " + name);
}

} // namespace quantsearch
