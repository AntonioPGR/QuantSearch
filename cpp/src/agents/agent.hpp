#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "../types.hpp"

namespace quantsearch {

using FeatureValue = std::variant<Vector, Matrix, Matrix3D>;
using AgentInput = std::map<std::string, FeatureValue>;

struct TrainingStats {
	double policy_loss = 0.0;
	double value_loss = 0.0;
};

class Agent {
public:
	virtual ~Agent() = default;
	virtual const std::string& name() const = 0;
	virtual Vector act(const AgentInput& input) = 0;
};

class TrainableAgent : public Agent {
public:
	virtual void setTraining(bool training) = 0;
	virtual void beginEpisode() = 0;
	virtual void recordReward(double reward, bool terminal) = 0;
	virtual TrainingStats updatePolicy() = 0;
};

std::unique_ptr<Agent> createAgent(const std::string& name, std::size_t stock_count);

} // namespace quantsearch
