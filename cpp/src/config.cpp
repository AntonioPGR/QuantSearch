#pragma once

struct Config {
	static constexpr int TRAIN_DAYS = 3 * 252;
	static constexpr int TEST_DAYS = 252;
	static constexpr int OBS_LEN = 50;
	static constexpr double STARTING_CASH = 10000.00;
	static constexpr double BANKRUPTCY_THRESHOLD = 0.0;
	static constexpr double RESET_CASH_THRESHOLD = 100000000.00; // testing money goes back to STARTING_CASH when both break or surpass
	static constexpr double MAX_DAILY_VALUE_CHANGE = 5000.00; // defense threshold for stock splits / abrupt price jumps

	static constexpr const char* DATA_DIR = "../_data";
	static constexpr const char* LOGS_DIR = "../_logs";
	static constexpr const char* RESULTS_DIR = "../_results";

	static constexpr int EPISODES_PER_WINDOW = 16;
	static constexpr bool WINNER_TAKES_ALL = false;

	static constexpr float NETWORK_THRESHOLD = 0.5;
};
