#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <vector>

#include "ppo.cpp"
#include "trading_env.cpp"
#include "config.cpp"

using namespace std;
using namespace torch;
using namespace torch::optim;
using namespace filesystem;


// LOG FILES BUFFER HANDLER CLASS ======================================================================================
class TeeBuffer : public streambuf {
	public:
		TeeBuffer(streambuf* terminal_buffer, streambuf* log_buffer)
			: terminal_buffer_(terminal_buffer), log_buffer_(log_buffer) {}

	protected:
		int_type overflow(int_type character) override {
			if (traits_type::eq_int_type(character, traits_type::eof())) return traits_type::not_eof(character);
			const auto terminal_result = terminal_buffer_->sputc(traits_type::to_char_type(character));
			const auto log_result = log_buffer_->sputc(traits_type::to_char_type(character));
			return traits_type::eq_int_type(terminal_result, traits_type::eof()) || traits_type::eq_int_type(log_result, traits_type::eof())
				? traits_type::eof() : character;
		}

		streamsize xsputn(const char* text, streamsize count) override {
			const streamsize terminal_count = terminal_buffer_->sputn(text, count);
			const streamsize log_count = log_buffer_->sputn(text, count);
			return min(terminal_count, log_count);
		}

		int sync() override {
			const int terminal_result = terminal_buffer_->pubsync();
			const int log_result = log_buffer_->pubsync();
			return terminal_result == 0 && log_result == 0 ? 0 : -1;
		}

	private:
		streambuf* terminal_buffer_;
		streambuf* log_buffer_;
};


// OUTPUT FILES HANDLER CLASS ==========================================================================================
struct OutputFiles {
	ofstream training_log;
	ofstream result_csv;
	unique_ptr<TeeBuffer> tee_buffer;
	streambuf* original_cout_buffer = nullptr;

	string timestamped_log_filename() {
		const auto now = chrono::system_clock::now();
		const auto milliseconds = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
		const time_t current_time = chrono::system_clock::to_time_t(now);
		stringstream filename;
		filename << put_time(localtime(&current_time), "%Y%m%d_%H%M%S") << '_' << setfill('0') << setw(3) << milliseconds;
		return filename.str();
	}

	OutputFiles(const vector<string>& stocks_names) {
		path workspace_root = filesystem::absolute(current_path() / "..");
		create_directories(workspace_root / "_logs");
		create_directories(workspace_root / "_results");

		string out_filename = timestamped_log_filename();

		training_log.open(workspace_root / "_logs" / (out_filename + ".txt"));
		result_csv.open(workspace_root / "_results" / (out_filename + ".csv"));

		if (!training_log || !result_csv) {
			throw runtime_error("Could not create training or test result CSV");
		}

		result_csv << "date,my_money,baseline_money";
		for (const string& stock_name : stocks_names) {
			result_csv << ",percent_stock_" << stock_name;
		}
		result_csv << '\n' << setprecision(12);

		tee_buffer = make_unique<TeeBuffer>(cout.rdbuf(), training_log.rdbuf());
		original_cout_buffer = cout.rdbuf(tee_buffer.get());

		if (Config::WINNER_TAKES_ALL) cout << "WINNER TAKES ALL = TRUE\n";
		cout << "WINNER TAKES ALL = FALSE\n";
	}

	~OutputFiles() {
		cout.flush();

		if (original_cout_buffer != nullptr) {
			cout.rdbuf(original_cout_buffer);
		}
	}
};

