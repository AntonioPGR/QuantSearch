#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class TradingEnv {
	public:
		static constexpr double BANKRUPTCY_REWARD = -1.0;

		TradingEnv(int train_days, int test_days){
			loadData();
			total_of_days = closes.size();
			window_train_days = train_days;
			window_test_days = test_days;
			reset(0, window_train_days);
		}

		void reset() {
			reset(window_start, window_end);
		}

		void reset(int start_day, int end_day, double starting_cash = 10000.0) {
			window_start = max(0, start_day);
			window_end = min(static_cast<int>(total_of_days), end_day);
			current_step = max(OBS_LEN - 1, window_start - 1);
			initial_cash = starting_cash;
			cash_portfolio = calculateInitialCashPortfolio();
			stock_portfolio = calculateStockPortfolioByCashPortfolio();
			percent_portfolio = calculatePercentPortfolioByCashPortfolio();
		}

		// Mercado fecha -> Recompensa -> observação -> ação -> próximo dia -> Mercado fecha -> Recompensa -> observação
		double step(vector<double>& action){
		if (terminated()) return BANKRUPTCY_REWARD;

			if (action.size() != static_cast<size_t>(stocks_amount)) return BANKRUPTCY_REWARD;
			double weight_sum = 0.0;
			for (double weight : action) {
				if (!isfinite(weight) || weight < 0.0) return BANKRUPTCY_REWARD;
				weight_sum += weight;
			}
			if (weight_sum <= 0.0) return BANKRUPTCY_REWARD;
			percent_portfolio = action;
			for (double& weight : percent_portfolio) weight /= weight_sum;
			cash_portfolio = calculateCashPortfolioByPercentPortfolio();
			stock_portfolio = calculateStockPortfolioByCashPortfolio();
			const double old_cash = calculateTotalCash();

			current_step++;

			cash_portfolio = calculateCashPortfolioByStockPortfolio();
			percent_portfolio = calculatePercentPortfolioByCashPortfolio();
			const double new_cash = calculateTotalCash();
			if (!isfinite(new_cash) || new_cash <= 0.0) {
				cash_portfolio.assign(stocks_amount, 0.0);
				return BANKRUPTCY_REWARD;
			}

			return std::log2(new_cash / old_cash);
		}

		vector<vector<double>> getPricesObservation(){
			auto res = std::vector<std::vector<double>>(OBS_LEN, std::vector<double>(stocks_amount, 0));
			const int start = current_step - OBS_LEN + 1;
			for (int i = 0; i < OBS_LEN; i++) {
				const int data_idx = start + i;
				for (int j = 0; j < stocks_amount; j++) {
					res[i][j] = closes[data_idx][j];
				}
			}
			return res;
		};

		vector<vector<double>> getVolumeObservation(){
			auto res = std::vector<std::vector<double>>(OBS_LEN, std::vector<double>(stocks_amount, 0));
			const int start = current_step - OBS_LEN + 1;
			for (int i = 0; i < OBS_LEN; i++) {
				const int data_idx = start + i;
				for (int j = 0; j < stocks_amount; j++) {
					res[i][j] = volumes[data_idx][j];
				}
			}
			return res;
		};

		vector<double> getPercentPortfolio(){
			return calculatePercentPortfolioByCashPortfolio();
		};

		const vector<string>& getStockNames() const { return stock_names; }
		const string& getCurrentDate() const { return dates[current_step]; }

		vector<double> getNextStepReturns() const {
			vector<double> returns(stocks_amount);
			for (int stock = 0; stock < stocks_amount; ++stock) {
				returns[stock] = closes[current_step + 1][stock] / closes[current_step][stock] - 1.0;
			}
			return returns;
		}

		size_t getBestPerformingStock(int start_day, int end_day) const {
			if (stocks_amount == 0 || start_day < 0 || end_day <= start_day || end_day > static_cast<int>(total_of_days)) {
				throw std::out_of_range("Invalid range for baseline stock selection");
			}
			size_t best_stock = 0;
			double best_return = closes[end_day - 1][0] / closes[start_day][0] - 1.0;
			for (int stock = 1; stock < stocks_amount; ++stock) {
				const double stock_return = closes[end_day - 1][stock] / closes[start_day][stock] - 1.0;
				if (stock_return > best_return) {
					best_return = stock_return;
					best_stock = static_cast<size_t>(stock);
				}
			}
			return best_stock;
		}

		bool terminated(){
			return current_step >= window_end - 1;
		};

		bool train_terminated(){
			return terminated();
		};

		bool has_days_remaining() const {
			return window_end > window_start + OBS_LEN;
		}

		double portfolio_value() const {
			double total = 0.0;
			for (double value : cash_portfolio) total += value;
			return total;
		}

		int first_usable_day() const { return OBS_LEN - 1; }
		int total_days() const { return static_cast<int>(total_of_days); }

		void setStep(int step) {
			this->current_step = step;
		}

	private:
		int CLOSE_COL = 1;
		int VOLUME_COL = 5;
		int OBS_LEN = 50;
		string DATA_DIR = "_data";

		int stocks_amount = 0;
		int current_step = 0;
		unsigned long total_of_days = 0;
		int window_train_days = 0;
		int window_test_days = 0;
		int window_start = 0;
		int window_end = 0;
		double initial_cash = 10000;
		vector<double> cash_portfolio;
		vector<double> stock_portfolio;
		vector<double> percent_portfolio;
		vector<vector<double>> closes;
		vector<vector<double>> volumes;
		vector<string> dates;
		vector<string> stock_names;

		void loadData(){
			closes.clear();
			volumes.clear();
			dates.clear();
			stock_names.clear();
			if (!std::filesystem::is_directory(DATA_DIR) && std::filesystem::is_directory("../_data")) {
				DATA_DIR = "../_data";
			} else if (!std::filesystem::is_directory(DATA_DIR) && std::filesystem::is_directory("../data")) {
				DATA_DIR = "../data";
			}
			if (!std::filesystem::is_directory(DATA_DIR)) {
				throw std::runtime_error("Could not find market data directory _data/ or ../_data/");
			}
			// FOR STOCK IN DIR
			stocks_amount = 0;
			for (const auto& entry : std::filesystem::directory_iterator(DATA_DIR)) {
				if (!entry.is_regular_file() || entry.path().extension() != ".csv") continue;
				std::ifstream file(entry.path().string());
				stock_names.push_back(entry.path().stem().string());
				// FOR LINE IN CSV
				std::string line, value;
				std::vector<std::string> values;
				std::getline(file, line); // jump first line
				int step_idx = 0;
				while (std::getline(file, line)) {
					if (line.empty()) continue;
					values.clear();
					std::stringstream ss(line);
					while (std::getline(ss, value, ',')) {
						values.push_back(value);
					}
					if (step_idx >= closes.size()) {
						closes.emplace_back();
						volumes.emplace_back();
						if (stocks_amount == 0) dates.emplace_back();
					}
					if (stocks_amount >= closes[step_idx].size()) {
						closes[step_idx].resize(stocks_amount + 1);
						volumes[step_idx].resize(stocks_amount + 1);
					}
					closes[step_idx][stocks_amount] = std::stod(values[CLOSE_COL]);
					volumes[step_idx][stocks_amount] = std::stod(values[VOLUME_COL]);
					if (stocks_amount == 0) dates[step_idx] = values[0];
					step_idx++;
				}
				stocks_amount++;
			}
		};

		vector<double> calculateInitialCashPortfolio(){
			auto arr = std::vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = initial_cash / stocks_amount;
			}
			return arr;
		};

		vector<double> calculateCashPortfolioByStockPortfolio(){
			auto arr = std::vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = stock_portfolio[i] * closes[current_step][i];
			}
			return arr;
		};

		vector<double> calculateCashPortfolioByPercentPortfolio(){
			double total = calculateTotalCash();
			auto arr = std::vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = total * percent_portfolio[i];
			}
			return arr;
		};

		vector<double> calculateStockPortfolioByCashPortfolio(){
			auto arr = std::vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = cash_portfolio[i] / closes[current_step][i];
			}
			return arr;
		};

		vector<double> calculatePercentPortfolioByCashPortfolio(){
			const double total_cash = calculateTotalCash();
			auto arr = std::vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = cash_portfolio[i] / total_cash;
			}
			return arr;
		};

		double calculateTotalCash(){
			double total = 0;
			for (int i = 0; i < stocks_amount; i++) {
				total += cash_portfolio[i];
			}
			return total;
		};
};

