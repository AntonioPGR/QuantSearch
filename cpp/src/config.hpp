#pragma once

#include <string_view>

struct Config {
	static constexpr int TRAIN_DAYS = 3 * 252;
	static constexpr int TEST_DAYS = 252;
	static constexpr int OBS_LEN = 50;
	static constexpr int EPISODES_PER_WINDOW = 16;

	static constexpr double STARTING_CASH = 10000.0;
	static constexpr double BANKRUPTCY_THRESHOLD = 0.0;
	static constexpr double RESET_CASH_THRESHOLD = 100000000.0;
	static constexpr double MAX_DAILY_VALUE_CHANGE = 5000.0;

	static constexpr int CLOSE_COLUMN = 1;
	static constexpr int VOLUME_COLUMN = 5;
	static constexpr double NETWORK_THRESHOLD = 0.5;
	static constexpr bool WINNER_TAKES_ALL = false;

	static constexpr double LEARNING_RATE = 3e-4;
	static constexpr double WEIGHT_DECAY = 1e-4;
	static constexpr int PPO_EPOCHS = 5;
	static constexpr int PPO_MINIBATCH_SIZE = 64;
	static constexpr double PPO_GAMMA = 0.99;
	static constexpr double PPO_LAMBDA = 0.95;
	static constexpr double PPO_CLIP_RANGE = 0.2;
	static constexpr double PPO_VALUE_COEF = 0.5;
	static constexpr double PPO_ENTROPY_COEF = 0.01;
	static constexpr double PPO_MAX_GRAD_NORM = 0.5;

	static constexpr const char* DATA_DIR = "_data";
	static constexpr const char* LOGS_DIR = "_logs";
	static constexpr const char* RESULTS_DIR = "_results";

	// Choose any subset of: price_matrix, volume_matrix, dual_matrix,
	// convolutional, convolutional_price, convolutional_volume,
	// same_percent, best_performing.
	static constexpr std::string_view ENABLED_AGENTS[] = {
		"convolutional", "convolutional_price", "convolutional_volume", "same_percent", "best_performing", "price_matrix", "volume_matrix", "dual_matrix"
	};
};
