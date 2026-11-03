#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "../types.hpp"

namespace quantsearch {

class TradingEnv {
public:
	TradingEnv();

	void reset(int start_step, int end_step);
	bool terminated() const;
	void advance();

	Matrix getPriceObservation() const;
	Matrix getVolumeObservation() const;
	const Vector& currentPrices() const;
	const Vector& nextPrices() const;
	const std::string& currentDate() const;
	const std::vector<std::string>& stockNames() const;
	int firstUsableDay() const;
	int totalDays() const;
	int currentStep() const;
	std::size_t stockCount() const;

private:
	void loadData();
	static std::filesystem::path findWorkspaceRoot();

	std::vector<std::string> dates_;
	std::vector<std::string> stock_names_;
	Matrix closes_;
	Matrix volumes_;
	int current_step_ = 0;
	int window_end_ = 0;
};

} // namespace quantsearch
