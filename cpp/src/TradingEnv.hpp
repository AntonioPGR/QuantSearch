#pragma once
#ifndef QUANTSEARCH_TRADINGENV_H
#define QUANTSEARCH_TRADINGENV_H

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
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

		void reset(int start_day, int end_day) {
			window_start = max(0, start_day);
			window_end = min(static_cast<int>(total_of_days), end_day);
			current_step = max(OBS_LEN - 1, window_start - 1);
			cash_portfolio = calculateInitialCashPortfolio();
			stock_portfolio = calculateStockPortfolioByCashPortfolio();
			percent_portfolio = calculatePercentPortfolioByCashPortfolio();
		}

		// Mercado fecha -> Recompensa -> observação -> ação -> próximo dia -> Mercado fecha -> Recompensa -> observação
		double step(vector<double>& action){
		if (terminated()) return BANKRUPTCY_REWARD;

			percent_portfolio = action;
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
		string DATA_DIR = "../data";

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

		void loadData(){
			closes.clear();
			volumes.clear();
			// FOR STOCK IN DIR
			stocks_amount = 0;
			for (const auto& entry : std::filesystem::directory_iterator(DATA_DIR)) {
				if (!entry.is_regular_file() || entry.path().extension() != ".csv") continue;
				std::ifstream file(entry.path().string());
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
					}
					if (stocks_amount >= closes[step_idx].size()) {
						closes[step_idx].resize(stocks_amount + 1);
						volumes[step_idx].resize(stocks_amount + 1);
					}
					closes[step_idx][stocks_amount] = std::stod(values[CLOSE_COL]);
					volumes[step_idx][stocks_amount] = std::stod(values[VOLUME_COL]);
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

#endif // QUANTSEARCH_TRADINGENV_H