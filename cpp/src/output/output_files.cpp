#include "output_files.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../config.hpp"

namespace quantsearch {

TeeBuffer::TeeBuffer(std::streambuf* terminal, std::streambuf* log) : terminal_(terminal), log_(log) {}

TeeBuffer::int_type TeeBuffer::overflow(int_type character) {
	if (traits_type::eq_int_type(character, traits_type::eof())) return traits_type::not_eof(character);
	const auto value = traits_type::to_char_type(character);
	const auto terminal_result = terminal_->sputc(value);
	const auto log_result = log_->sputc(value);
	return traits_type::eq_int_type(terminal_result, traits_type::eof()) ||
	       traits_type::eq_int_type(log_result, traits_type::eof()) ? traits_type::eof() : character;
}

std::streamsize TeeBuffer::xsputn(const char* text, std::streamsize count) {
	return std::min(terminal_->sputn(text, count), log_->sputn(text, count));
}

int TeeBuffer::sync() {
	return terminal_->pubsync() == 0 && log_->pubsync() == 0 ? 0 : -1;
}

std::filesystem::path OutputFiles::findWorkspaceRoot() {
	auto directory = std::filesystem::current_path();
	for (int depth = 0; depth < 4; ++depth) {
		if (std::filesystem::is_directory(directory / Config::DATA_DIR)) return directory;
		if (!directory.has_parent_path() || directory.parent_path() == directory) break;
		directory = directory.parent_path();
	}
	throw std::runtime_error("Could not locate project directory containing '" + std::string(Config::DATA_DIR) + "'");
}

std::string OutputFiles::timestamp() {
	const auto now = std::chrono::system_clock::now();
	const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
	const std::time_t current_time = std::chrono::system_clock::to_time_t(now);
	std::stringstream result;
	result << std::put_time(std::localtime(&current_time), "%Y%m%d_%H%M%S")
	       << '_' << std::setfill('0') << std::setw(3) << milliseconds;
	return result.str();
}

OutputFiles::OutputFiles(const std::vector<std::string>& stock_names, const std::vector<std::string>& agent_names)
	: agent_names_(agent_names), stock_count_(stock_names.size()) {
	if (agent_names_.empty()) throw std::invalid_argument("At least one agent must be enabled");
	const auto root = findWorkspaceRoot();
	std::filesystem::create_directories(root / Config::LOGS_DIR);
	std::filesystem::create_directories(root / Config::RESULTS_DIR);
	const std::string file_name = timestamp();
	training_log.open(root / Config::LOGS_DIR / (file_name + ".txt"));
	result_csv.open(root / Config::RESULTS_DIR / (file_name + ".csv"));
	if (!training_log || !result_csv) throw std::runtime_error("Could not create experiment output files");

	result_csv << "date";
	for (const auto& agent_name : agent_names_) result_csv << ',' << agent_name << "_money";
	for (const auto& agent_name : agent_names_) {
		for (const auto& stock_name : stock_names) result_csv << ',' << agent_name << "_percent_stock_" << stock_name;
	}
	result_csv << '\n' << std::setprecision(12);

	tee_buffer_ = std::make_unique<TeeBuffer>(std::cout.rdbuf(), training_log.rdbuf());
	original_cout_buffer_ = std::cout.rdbuf(tee_buffer_.get());
}

OutputFiles::~OutputFiles() {
	std::cout.flush();
	if (original_cout_buffer_ != nullptr) std::cout.rdbuf(original_cout_buffer_);
}

void OutputFiles::writeResults(const std::string& date, const std::vector<AgentResult>& results) {
	if (results.size() != agent_names_.size()) throw std::invalid_argument("Result count does not match enabled agents");
	result_csv << date;
	for (const auto& result : results) result_csv << ',' << result.money;
	for (const auto& result : results) {
		if (result.weights.size() != stock_count_) throw std::invalid_argument("Agent returned a portfolio with the wrong number of weights");
		for (double weight : result.weights) result_csv << ',' << weight * 100.0;
	}
	result_csv << '\n';
}

} // namespace quantsearch
