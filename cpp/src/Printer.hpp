#pragma once
#ifndef QUANTSEARCH_PRINTER_H
#define QUANTSEARCH_PRINTER_H

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

class Printer {
	public:

		static void printMatrix(const vector<vector<double>>& matrix) {
			for (const auto& row : matrix) {
				for (double value : row) {
					if (value < 0) cout << fixed << setprecision(4) << value << " | ";
					else cout << fixed << setprecision(4) << " " << value << " | ";
				}
				cout << '\n';
			}
		}

		static void printLine() {
			std::cout << "----------------------------------------" << '\n';
		};
};

#endif //QUANTSEARCH_PRINTER_H
