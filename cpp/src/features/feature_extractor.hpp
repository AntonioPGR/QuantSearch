#pragma once

#include "../agents/agent.hpp"

namespace quantsearch {

class FeatureExtractor {
public:
	static AgentInput extract(const Matrix& prices, const Matrix& volumes);
};

} // namespace quantsearch
