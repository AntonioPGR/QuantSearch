#include <cmath>
#include <vector>

using namespace std;

class BaselineCalculator {

	public:
		// stocks_ts -> stock price x time [[price t, price t+1, price t+2], [price t, price t+1, price t+2]]
		static vector<double> calculateBestStockBaseline(vector<vector<double>> stocks_ts) {
			int n = stocks_ts.size();
			vector<double> res = vector<double>(n, 0);
			int best_idx = 0;
			double best_value = 0;
			for (int i = 0; i < stocks_ts.size(); i++) {
				vector<double> stock = stocks_ts[i];
				double sumOfLogs = 0;
				for (int j = 1; j < stock.size(); j++) sumOfLogs += log(stock[j] / stock[j-1]);
				double mean = sumOfLogs / stock.size();
				if (mean > best_value) {
					best_value = mean;
					best_idx = i;
				}
			}
			res[best_idx] = 1;
			return res;
		}

		// stocks_ts -> stock price x time [[price t, price t+1, price t+2], [price t, price t+1, price t+2]]
		static vector<double> calculateSamePercentBaseline(vector<vector<double>> stocks_ts) {
			int n = stocks_ts.size();
			vector<double> percents = vector<double>(n, 100 / n);
			int sum = 0;
			for (double v : percents) sum += v;
			if (sum < 100) percents[n-1] += 100 - sum;
			if (sum > 100) percents[n-1] -= sum - 100;
			return percents;
		}

};
