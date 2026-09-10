#ifndef QUANTSEARCH_TRADINGENV_H
#define QUANTSEARCH_TRADINGENV_H

#include <bits/stdc++.h>

using namespace std;

class TradingEnv {

	public:
		TradingEnv() {
			loadData();
			total_of_days = closes.size();
			reset();
		}

		void reset() {
			current_step = OBS_LEN-1;
			cash_portfolio = calculateInitialCashPortfolio();
			stock_portfolio = calculateStockPortfolioByCashPortfolio();
			percent_portfolio = calculatePercentPortfolioByCashPortfolio();
		}

		// Mercado fecha -> observação -> ação -> próximo dia -> Mercado fecha -> Recompensa -> observação
		double step(const vector<double>& action) {
			percent_portfolio = action;
			cash_portfolio = calculateCashPortfolioByPercentPortfolio();
			stock_portfolio = calculateStockPortfolioByCashPortfolio();
			const double old_cash = calculateTotalCash();

			current_step++;

			cash_portfolio = calculateCashPortfolioByStockPortfolio();
			percent_portfolio = calculatePercentPortfolioByCashPortfolio();
			const double new_cash = calculateTotalCash();

			return log2(new_cash / old_cash);
		}

		[[nodiscard]] vector<vector<double>> getObservation() const {
			auto res = vector<vector<double>>(OBS_LEN, vector<double>(stocks_amount * 2, 0));
			const int start = current_step - OBS_LEN + 1;
			for (int i = 0; i < OBS_LEN; i++) {
				const int data_idx = start + i;
				for (int j = 0; j < stocks_amount; j++) {
					res[i][j] = closes[data_idx][j];
					res[i][stocks_amount + j] = volumes[data_idx][j];
				}
			}
			return res;
		}

		[[nodiscard]] bool terminated() const {
			return current_step >= OBS_LEN-1;
		}


	private:
		int CLOSE_COL = 1;
		int VOLUME_COL = 5;
		int OBS_LEN = 50;
		string DATA_DIR = "_DATA";

		int stocks_amount = 0;
		int current_step = 0;
		unsigned long total_of_days = 0;
		double initial_cash = 10000;
		vector<double> cash_portfolio = vector<double>();
		vector<double> stock_portfolio = vector<double>();
		vector<double> percent_portfolio = vector<double>();
		vector<vector<double>> closes = vector<vector<double>>(); // [step][stock]
		vector<vector<double>> volumes = vector<vector<double>>();


		void loadData() {
			closes.clear();
			volumes.clear();
			// FOR STOCK IN DIR
			stocks_amount = 0;
			for (const auto& entry : filesystem::directory_iterator(DATA_DIR)){
				if (!entry.is_regular_file() || entry.path().extension() != ".csv") continue;
				ifstream file(entry.path().string());
				// FOR LINE IN CSV
				string line, value;
				vector<string> values;
				getline(file, line); // jump first line
				int step_idx = 0;
				while (getline(file, line)) {
					if (line.empty()) continue;
					values.clear();
					stringstream ss(line);
					while (getline(ss, value, ',')) {
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
					closes[step_idx][stocks_amount] = stod(values[CLOSE_COL]);
					volumes[step_idx][stocks_amount] = stod(values[VOLUME_COL]);
					step_idx++;
				}
				stocks_amount++;
			}
		}

		[[nodiscard]] vector<double> calculateInitialCashPortfolio() const {
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = initial_cash / stocks_amount;
			}
			return arr;
		}

		[[nodiscard]] vector<double> calculateCashPortfolioByStockPortfolio() const {
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = stock_portfolio[i] * closes[current_step][i];
			}
			return arr;
		}

		[[nodiscard]] vector<double> calculateCashPortfolioByPercentPortfolio() const {
			double total = calculateTotalCash();
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = total * percent_portfolio[i];
			}
			return arr;
		}

		[[nodiscard]] vector<double> calculateStockPortfolioByCashPortfolio() const {
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = cash_portfolio[i] / closes[current_step][i];
			}
			return arr;
		}

		[[nodiscard]] vector<double> calculatePercentPortfolioByCashPortfolio() const {
			const double total_cash = calculateTotalCash();
			auto arr = vector<double>(stocks_amount, 0);
			for (int i = 0; i < stocks_amount; i++) {
				arr[i] = cash_portfolio[i] / total_cash;
			}
			return arr;
		}

		[[nodiscard]] double calculateTotalCash() const {
			double total = 0;
			for (int i = 0; i < stocks_amount; i++) {
				total += cash_portfolio[i];
			}
			return total;
		}

};

#endif // QUANTSEARCH_TRADINGENV_H