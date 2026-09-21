#pragma once
#ifndef QUANTSEARCH_NETWORKBUILDER_H
#define QUANTSEARCH_NETWORKBUILDER_H

#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

class NetworkBuilder {

	float base_threshold = 0.5;

	public:
		vector<vector<double>> build(const vector<vector<double>>& obs){
			vector<vector<double>> correlations = calculateCorrelationMatrix(obs);
			int number_of_stocks = correlations[0].size();

			if (number_of_stocks < 2) {
				return vector<vector<double>>(number_of_stocks, vector<double>(number_of_stocks, 0.0));
			}

			vector<double> absolute_values;
			for (int i = 0; i < number_of_stocks; i++) {
				for (int j = i + 1; j < number_of_stocks; j++) {
					absolute_values.push_back(abs(correlations[i][j]));
				}
			}
			int edge_count = absolute_values.size();
			int number_to_keep = max(1, edge_count / 2);
			sort(absolute_values.begin(), absolute_values.end());
			double percent_threshold = absolute_values[edge_count - number_to_keep];

			vector<vector<double>> graph = correlations;
			for (int i = 0; i < number_of_stocks; i++) {
				for (int j = 0; j < number_of_stocks; j++) {
					if (abs(graph[i][j]) < percent_threshold || abs(graph[i][j]) < base_threshold) graph[i][j] = 0.0;
				}
			}

			return graph;
		};

	private:
		vector<vector<double>> calculateCorrelationMatrix(const vector<vector<double>>& obs){
			int timesteps = obs.size();
			int stocks = obs[0].size();

			vector<vector<double>> correlation(stocks, vector<double>(stocks, 0.0));
			vector<double> means(stocks, 0.0);

			for (int stock = 0; stock < stocks; stock++) {
				for (int t = 0; t < timesteps; t++) {
					means[stock] += obs[t][stock];
				}
				means[stock] /= timesteps;
			}

			for (int i = 0; i < stocks; i++) {
				for (int j = i; j < stocks; j++) {
					double numerator = 0.0;
					double sum_i = 0.0;
					double sum_j = 0.0;
					for (int t = 0; t < timesteps; t++) {
						double diff_i = obs[t][i] - means[i];
						double diff_j = obs[t][j] - means[j];
						numerator += diff_i * diff_j;
						sum_i += diff_i * diff_i;
						sum_j += diff_j * diff_j;
					}
					double denominator = sqrt(sum_i * sum_j);
					double corr = denominator == 0.0 ? 0.0 : numerator / denominator;
					correlation[i][j] = corr;
					correlation[j][i] = corr;
				}
			}

			return correlation;
		};

};

#endif //QUANTSEARCH_NETWORKBUILDER_H
