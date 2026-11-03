#include <exception>
#include <iostream>

#include "agents/experiment_runner.hpp"

int main() {
	try {
		quantsearch::runExperiments();
	} catch (const std::exception& error) {
		std::cerr << "QuantSearch failed: " << error.what() << '\n';
		return 1;
	}
	return 0;
}
