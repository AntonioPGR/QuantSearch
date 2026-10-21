#pragma once

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.cpp"

using namespace std;
using namespace std::filesystem;

// ASSUNÇÕES --> TODAS AS AÇÕES ESTÃO INDEXADAS POR DATA E POSSUEM AS MESMAS DATAS NAS MESMAS LINHAS

// CONSTANTS USED IN THE SHAPES BELOW ==================================================================================
// N = Number of stocks (amount of assets in the portfolio, stocks_amount)
// D = Total of days in the data (total_of_days)
// O = Observation length (OBS_LEN, amount of past days used to build the financial networks)
//
// closes / volumes      -> [D][N]
// cash_portfolio        -> [N] money in each stock
// stock_portfolio       -> [N] amount of shares of each stock
// percent_portfolio     -> [N] cash_portfolio / total cash (sums to 1, or all 0 when bankrupt)
// prices / volume obs   -> [O][N] days [current_step - O + 1, current_step]
//
// WINDOW TIMELINE =====================================================================================================
// reset(start, end) -> current_step = start - 1 (the agent decides at the close of the day before the window)
// every step()      -> realizes the return of day current_step + 1, so the window realizes days [start, end - 1]
// terminated()      -> current_step == end - 1

class TradingEnv {
	public:
		static constexpr double BANKRUPTCY_REWARD = -1.0;

		TradingEnv(){
			loadData();
			resetTraining(firstUsableDay(), total_of_days);
		}

		void resetTraining(int startStep, int endStep) {
			reset(startStep, endStep, Config::STARTING_CASH);
		}
		// end_step is exclusive: test_start for the training window, test_end for the testing window
		// portfolio is the money the window starts with: STARTING_CASH when training, the last testing money when testing
		void reset(int start_step, int end_step, double portfolio = Config::STARTING_CASH) {
			if (start_step < OBS_LEN) throw runtime_error("TradingEnv -> Reset: Invalid start step, cannot be smaller than OBS_LEN");
			if (end_step <= start_step || end_step > static_cast<int>(total_of_days)) throw runtime_error("TradingEnv -> Reset: Invalid end step, must be in (start step, total of days]");
			window_start = start_step;
			window_end = end_step;
			current_step = window_start - 1;
			resetMoney(portfolio);
		}

		// Keeps the window and the current step, only replaces the money (split equally between the N stocks)
		void resetMoney(double portfolio = Config::STARTING_CASH) {
			cash_portfolio = calculateInitialCashPortfolio(portfolio);
			stock_portfolio = calculateStockPortfolioByCashPortfolio();
			percent_portfolio = calculatePercentPortfolioByCashPortfolio();
		}

		// Mercado fecha -> Recompensa -> observação -> ação -> próximo dia -> Mercado fecha -> Recompensa -> observação
		double step(const vector<double>& action){
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
			double new_cash = calculateTotalCash();

			// Split defense: if portfolio value varies abruptly by more than MAX_DAILY_VALUE_CHANGE, keep last day's value and redistribute actions
			if (abs(new_cash - old_cash) > Config::MAX_DAILY_VALUE_CHANGE) {
				for (int i = 0; i < stocks_amount; i++) {
					cash_portfolio[i] = old_cash * percent_portfolio[i];
				}
				stock_portfolio = calculateStockPortfolioByCashPortfolio();
				percent_portfolio = calculatePercentPortfolioByCashPortfolio();
				new_cash = old_cash;
			}

			if (!isfinite(new_cash) || new_cash <= Config::BANKRUPTCY_THRESHOLD) {
				cash_portfolio.assign(stocks_amount, 0.0);
				return BANKRUPTCY_REWARD;
			}

			return log2(new_cash / old_cash);
		}


		vector<vector<double>> getPricesObservation(){
			auto res = vector<vector<double>>(OBS_LEN, vector<double>(stocks_amount, 0));
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
			auto res = vector<vector<double>>(OBS_LEN, vector<double>(stocks_amount, 0));
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

		vector<string> getStockNames() const { return stock_names; }
		const string& getCurrentDate() const { return dates[current_step]; }

		// Baseline buy / sell: buys `stock` with all the cash at the close of current_step and sells at the close of current_step + 1
		// Must be called BEFORE step(), so it realizes the same day the agent realizes
		double baselineBuyAndSell(double cash, size_t stock) const {
			const double shares = cash / closes[current_step][stock];
			double new_cash = shares * closes[current_step + 1][stock];
			if (abs(new_cash - cash) > Config::MAX_DAILY_VALUE_CHANGE) {
				new_cash = cash;
			}
			return new_cash;
		}

		// Best stock over the same O days the agent observes: [current_step - O + 1, current_step]
		size_t getBestPerformingStock() const {
			const int start_day = static_cast<int>(current_step) - OBS_LEN + 1;
			const int end_day = static_cast<int>(current_step) + 1;
			if (stocks_amount == 0 || start_day < 0 || end_day <= start_day || end_day > static_cast<int>(total_of_days)) {
				throw out_of_range("Invalid range for baseline stock selection");
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

		// The first decision is taken at current_step = start - 1, which needs O days [start - O, start - 1]
		int firstUsableDay() const { return OBS_LEN; }
		void setStep(int step) {this->current_step = step;}

		int getTotalDays() const {
			return total_of_days;
		}
		int getStocksAmount() const {
			return stocks_amount;
		}

	private:
		// FIXED VARS
		int CLOSE_COL = 1;
		int VOLUME_COL = 5;

		// CONSTRUCTOR VARS
		int OBS_LEN = Config::OBS_LEN;
		string DATA_DIR = Config::DATA_DIR;
		int TRAIN_DAYS = Config::TRAIN_DAYS;
		int TEST_DAYS = Config::TEST_DAYS;
		double STARTING_CASH = Config::STARTING_CASH;

		// CALCULATED VARS
		unsigned int total_of_days = 0;
		unsigned int stocks_amount = 0;
		unsigned int current_step = 0;
		unsigned int window_start = 0;
		unsigned int window_end = 0;

		// DATA VECTORS
		vector<double> cash_portfolio;
		vector<double> stock_portfolio;
		vector<double> percent_portfolio;
		vector<string> dates;
		vector<string> stock_names;

		// DATA MATRIXS
		vector<vector<double>> closes;
		vector<vector<double>> volumes;

		void loadData(){
			closes.clear();
			volumes.clear();
			dates.clear();
			stock_names.clear();

			if (!is_directory(DATA_DIR)) throw runtime_error("Could not find market data directory for path " + DATA_DIR);

			// FOR STOCK IN DIR
			stocks_amount = 0;
			for (const auto& entry : directory_iterator(DATA_DIR)) {
				path file_path = entry.path();
				if (!entry.is_regular_file() || file_path.extension() != ".csv") continue;
				ifstream file(file_path.string());
				string line, value;
				vector<string> values;
				getline(file, line); // jump first line
				int step_idx = 0;
				while (getline(file, line)) {
					if (line.empty()) continue;
					values.clear();
					stringstream ss(line);
					while (getline(ss, value, ',')) values.push_back(value);
					if (step_idx >= closes.size()) {
						closes.emplace_back();
						volumes.emplace_back();
						if (stocks_amount == 0) dates.emplace_back();
					}
					if (stocks_amount >= closes[step_idx].size()) {
						closes[step_idx].resize(stocks_amount + 1);
						volumes[step_idx].resize(stocks_amount + 1);
					}
					closes[step_idx][stocks_amount] = stod(values[CLOSE_COL]);
					volumes[step_idx][stocks_amount] = stod(values[VOLUME_COL]);
					if (stocks_amount == 0) dates[step_idx] = values[0];
					step_idx++;
				}
				stock_names.push_back(file_path.stem().string());
				stocks_amount++;
			}

			total_of_days = closes.size();
		};

		vector<double> calculateInitialCashPortfolio(double portfolio){
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = portfolio / stocks_amount;
			}
			return arr;
		};

		vector<double> calculateCashPortfolioByStockPortfolio(){
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = stock_portfolio[i] * closes[current_step][i];
			}
			return arr;
		};

		vector<double> calculateCashPortfolioByPercentPortfolio(){
			double total = calculateTotalCash();
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = total * percent_portfolio[i];
			}
			return arr;
		};

		vector<double> calculateStockPortfolioByCashPortfolio(){
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = cash_portfolio[i] / closes[current_step][i];
			}
			return arr;
		};

		vector<double> calculatePercentPortfolioByCashPortfolio(){
			const double total_cash = calculateTotalCash();
			auto arr = vector<double>(stocks_amount, 0);
			if (total_cash <= Config::BANKRUPTCY_THRESHOLD) return arr; // bankrupt -> all 0 instead of 0 / 0 = NaN
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

