#pragma once

struct Config {
	static constexpr int TRAIN_DAYS = 3 * 252;
	static constexpr int TEST_DAYS = 252;
	static constexpr int OBS_LEN = 50;
	static constexpr double STARTING_CASH = 10000.00;

	static constexpr const char* DATA_DIR = "../_data";
	static constexpr const char* LOGS_DIR = "../_logs";
	static constexpr const char* RESULTS_DIR = "../_results";

	static constexpr int EPISODES_PER_WINDOW = 16;
	static constexpr bool WINNER_TAKES_ALL = false;
};
