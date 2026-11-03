#include "trading_env.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "../config.hpp"

namespace quantsearch {

namespace {

std::vector<std::string> splitCsvLine(const std::string& line) {
	std::vector<std::string> fields;
	std::stringstream stream(line);
	std::string field;
	while (std::getline(stream, field, ',')) fields.push_back(field);
	return fields;
}

} // namespace

TradingEnv::TradingEnv() {
	loadData();
}

std::filesystem::path TradingEnv::findWorkspaceRoot() {
	auto directory = std::filesystem::current_path();
	for (int depth = 0; depth < 4; ++depth) {
		if (std::filesystem::is_directory(directory / Config::DATA_DIR)) return directory;
		if (!directory.has_parent_path() || directory.parent_path() == directory) break;
		directory = directory.parent_path();
	}
	throw std::runtime_error("Could not locate market data directory '" + std::string(Config::DATA_DIR) + "'");
}

void TradingEnv::loadData() {
	const auto data_directory = findWorkspaceRoot() / Config::DATA_DIR;
	std::vector<std::filesystem::path> files;
	for (const auto& entry : std::filesystem::directory_iterator(data_directory)) {
		if (entry.is_regular_file() && entry.path().extension() == ".csv") files.push_back(entry.path());
	}
	std::sort(files.begin(), files.end());
	if (files.empty()) throw std::runtime_error("No stock CSV files found in " + data_directory.string());

	for (const auto& file_path : files) {
		std::ifstream file(file_path);
		if (!file) throw std::runtime_error("Could not open stock data file " + file_path.string());

		std::string line;
		if (!std::getline(file, line)) throw std::runtime_error("Stock data file is empty: " + file_path.string());
		std::vector<double> closes;
		std::vector<double> volumes;
		std::vector<std::string> dates;
		while (std::getline(file, line)) {
			if (line.empty()) continue;
			const auto fields = splitCsvLine(line);
			if (fields.size() <= static_cast<std::size_t>(std::max(Config::CLOSE_COLUMN, Config::VOLUME_COLUMN))) {
				throw std::runtime_error("Malformed CSV row in " + file_path.string());
			}
			const double close = std::stod(fields[Config::CLOSE_COLUMN]);
			const double volume = std::stod(fields[Config::VOLUME_COLUMN]);
			if (!std::isfinite(close) || close <= 0.0 || !std::isfinite(volume) || volume < 0.0) {
				throw std::runtime_error("Invalid close or volume value in " + file_path.string());
			}
			dates.push_back(fields[0]);
			closes.push_back(close);
			volumes.push_back(volume);
		}

		if (stock_names_.empty()) {
			dates_ = std::move(dates);
			closes_.resize(dates_.size());
			volumes_.resize(dates_.size());
		} else if (dates != dates_) {
			throw std::runtime_error("Stock CSV files do not contain the same dates: " + file_path.string());
		}
		if (closes.size() != dates_.size()) {
			throw std::runtime_error("Stock CSV files have different numbers of rows: " + file_path.string());
		}

		for (std::size_t day = 0; day < dates_.size(); ++day) {
			closes_[day].push_back(closes[day]);
			volumes_[day].push_back(volumes[day]);
		}
		stock_names_.push_back(file_path.stem().string());
	}
	if (dates_.size() <= static_cast<std::size_t>(Config::OBS_LEN)) {
		throw std::runtime_error("Market data has too few rows for the configured observation length");
	}
}

void TradingEnv::reset(int start_step, int end_step) {
	if (start_step < firstUsableDay() || end_step <= start_step || end_step > totalDays()) {
		throw std::invalid_argument("Invalid trading window boundaries");
	}
	current_step_ = start_step - 1;
	window_end_ = end_step;
}

bool TradingEnv::terminated() const {
	return current_step_ >= window_end_ - 1;
}

void TradingEnv::advance() {
	if (terminated()) throw std::logic_error("Cannot advance a terminated trading window");
	++current_step_;
}

Matrix TradingEnv::getPriceObservation() const {
	const int start = current_step_ - Config::OBS_LEN + 1;
	Matrix observation(Config::OBS_LEN, Vector(stockCount()));
	for (int day = 0; day < Config::OBS_LEN; ++day) {
		observation[day] = closes_[start + day];
	}
	return observation;
}

Matrix TradingEnv::getVolumeObservation() const {
	const int start = current_step_ - Config::OBS_LEN + 1;
	Matrix observation(Config::OBS_LEN, Vector(stockCount()));
	for (int day = 0; day < Config::OBS_LEN; ++day) {
		observation[day] = volumes_[start + day];
	}
	return observation;
}

const Vector& TradingEnv::currentPrices() const {
	return closes_.at(static_cast<std::size_t>(current_step_));
}

const Vector& TradingEnv::nextPrices() const {
	return closes_.at(static_cast<std::size_t>(current_step_ + 1));
}

const std::string& TradingEnv::currentDate() const {
	return dates_.at(static_cast<std::size_t>(current_step_));
}

const std::vector<std::string>& TradingEnv::stockNames() const {
	return stock_names_;
}

int TradingEnv::firstUsableDay() const {
	return Config::OBS_LEN;
}

int TradingEnv::totalDays() const {
	return static_cast<int>(dates_.size());
}

int TradingEnv::currentStep() const {
	return current_step_;
}

std::size_t TradingEnv::stockCount() const {
	return stock_names_.size();
}

} // namespace quantsearch
