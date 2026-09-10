#include <bits/stdc++.h>
#include "TradingEnv.h"

using namespace std;

int main() {
	TradingEnv env = TradingEnv();

	const auto observation = env.getObservation();

	cout << fixed << setprecision(2);

	cout << "First Observation\n";
	cout << string(100, '-') << '\n';

	for (size_t i = 0; i < observation.size(); i++) {
		cout << "Step " << i << " | ";

		for (size_t j = 0; j < observation[i].size(); j++) {
			cout << observation[i][j];

			if (j + 1 < observation[i].size()) {
				cout << " | ";
			}
		}

		cout << '\n';
	}

	cout << string(100, '-') << '\n';

	return 0;
}