#pragma once

#include "../types.hpp"

namespace quantsearch {

class NetworkFeatureExtractor {
public:
	static constexpr std::size_t nodeFeatureCount = 8;
	static constexpr std::size_t linkFeatureCount = 8;

	static Matrix extractNodeLevelFeatures(const Matrix& graph);
	static Matrix3D extractLinkLevelFeatures(const Matrix& graph);
};

} // namespace quantsearch
