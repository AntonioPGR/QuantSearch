#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <streambuf>
#include <string>
#include <vector>

#include "../agents/agent.hpp"

namespace quantsearch {

struct AgentResult {
	double money = 0.0;
	Vector weights;
};

class TeeBuffer final : public std::streambuf {
public:
	TeeBuffer(std::streambuf* terminal, std::streambuf* log);

protected:
	int_type overflow(int_type character) override;
	std::streamsize xsputn(const char* text, std::streamsize count) override;
	int sync() override;

private:
	std::streambuf* terminal_;
	std::streambuf* log_;
};

class OutputFiles {
public:
	OutputFiles(const std::vector<std::string>& stock_names, const std::vector<std::string>& agent_names);
	~OutputFiles();

	void writeResults(const std::string& date, const std::vector<AgentResult>& results);

	std::ofstream training_log;
	std::ofstream result_csv;

private:
	static std::filesystem::path findWorkspaceRoot();
	static std::string timestamp();

	std::unique_ptr<TeeBuffer> tee_buffer_;
	std::streambuf* original_cout_buffer_ = nullptr;
	std::vector<std::string> agent_names_;
	std::size_t stock_count_ = 0;
};

} // namespace quantsearch
