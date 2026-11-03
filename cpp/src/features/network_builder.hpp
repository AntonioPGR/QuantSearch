#pragma once

#include "../types.hpp"

namespace quantsearch {

class NetworkBuilder {
public:
	static Matrix build(const Matrix& observations);
	static Matrix calculateCorrelationMatrix(const Matrix& observations);
};

} // namespace quantsearch
