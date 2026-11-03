#pragma once

#include "agent.hpp"

namespace quantsearch {

class SamePercentAgent final : public Agent {
public:
	explicit SamePercentAgent(std::size_t stock_count);
	const std::string& name() const override;
	Vector act(const AgentInput& input) override;

private:
	std::string name_ = "same_percent";
	std::size_t stock_count_;
};

class BestPerformingAgent final : public Agent {
public:
	explicit BestPerformingAgent(std::size_t stock_count);
	const std::string& name() const override;
	Vector act(const AgentInput& input) override;

private:
	std::string name_ = "best_performing";
	std::size_t stock_count_;
};

} // namespace quantsearch
