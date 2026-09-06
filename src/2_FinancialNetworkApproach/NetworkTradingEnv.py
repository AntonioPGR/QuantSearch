from typing import SupportsFloat, Any
import gymnasium as gym
import pandas as pd
from gymnasium.core import ActType, ObsType
import numpy as np
from gymnasium import spaces

# SETTINGS ============================
SEED = 42
STOCKS = ["SBSP3", "BBDC3", "BRAP4", "VALE3", "GGBR4", "ISAE4", "CSNA3", "CMIG4", "EMBJ3", "CPLE3", "USIM5", "ITSA4",
	"AXIA3", "VIVT3", "WEGE3", "POMO4", "PETR4", "PETR3"]
STOCKS_LEN = len(STOCKS)
OBS_LEN = 50
INITIAL_CASH = 10_000.0
DATA_DIR = "data"


# ENV =================================
class NetworkTradingEnv(gym.Env):
	metadata = {"render_modes": []}

	def __init__(self):
		super().__init__()
		self.close_prices = self._loadClosePrices()
		self.total_of_days = len(self.close_prices)

		self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(OBS_LEN, STOCKS_LEN), dtype=np.float32)
		self.action_space = spaces.Box(low=0.0, high=1.0, shape=(STOCKS_LEN,), dtype=np.float32)

		self.current_step = OBS_LEN
		self.cash_portfolio = []
		self.stock_portfolio = []
		self.percent_portfolio = []
		self.total_cash = INITIAL_CASH
		
		self.reset()

	# ENV FUNCTIONS =================================
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
		super().reset(seed=seed if seed is not None else SEED)

		self.current_step = OBS_LEN
		self.cash_portfolio = self._calculateInitialCashPortfolio()
		self.total_cash = self._calculateTotalCashByCashPortfolio()
		self.stock_portfolio = self._calculateStockPortfolioByCashPortfolio()
		self.percent_portfolio = self._calculatePercentPortfolioByCashPortfolio()

		observation = self._getObservation()
		info = self._getInfo()
		return observation, info

	def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		action_sum = np.sum(action)
		if action_sum == 0 or np.isnan(action_sum):
			self.percent_portfolio = [1.0 / STOCKS_LEN] * STOCKS_LEN
		else:
			self.percent_portfolio = [float(a / action_sum) for a in action]

		self.cash_portfolio = self._calculateCashPortfolioByPercentPortfolio()
		self.stock_portfolio = self._calculateStockPortfolioByCashPortfolio()
		old_total_cash = self.total_cash

		self.current_step += 1
		terminated = self.current_step >= self.total_of_days

		if not terminated:
			self.cash_portfolio = self._calculateCashPortfolioByStockPortfolio()
			self.total_cash = self._calculateTotalCashByCashPortfolio()
			self.percent_portfolio = self._calculatePercentPortfolioByCashPortfolio()

		reward = self._calculateReward(old_total_cash, self.total_cash)
		truncated = False
		observation = self._getObservation()
		info = self._getInfo()

		return observation, reward, terminated, truncated, info
	
	def render(self) -> None:
		items = self._getInfo()
		for key, value in items.items():
			if isinstance(value, dict):
				for k, v in value.items():
					print(f"{k}: {f'{v:.3f}' if isinstance(v, float) else v}")
			elif isinstance(value, list):
				formatted_list = [f"{v:.3f}" if isinstance(v, float) else v for v in value]
				print(f"{key}: {formatted_list}")
			elif isinstance(value, float):
				print(f"{key}: {value:.3f}")
			else:
				print(f"{key}: {value}")

	# HELPER FUNCTIONS =================================
	@staticmethod
	def _loadClosePrices() -> list[list[float]]:
		series_list = []
		lengths = []
		for stock in STOCKS:
			df = pd.read_csv(f"{DATA_DIR}/{stock}.csv", usecols=["Close"])
			close_series = pd.to_numeric(df["Close"], errors="coerce")
			np_arr = close_series.to_numpy(dtype=np.float64)
			py_list = [float(x) for x in np_arr]
			series_list.append(py_list)
			lengths.append(len(py_list))
		days = lengths[0]
		prices = [[series_list[stock_idx][day_idx] for stock_idx in range(STOCKS_LEN)] for day_idx in range(days)]
		return prices

	def _getObservation(self) -> np.ndarray:
		# Extract exactly OBS_LEN timesteps ending at current_step
		start = self.current_step - OBS_LEN
		end = self.current_step
		window = [self.close_prices[i] for i in range(start, end)]
		return np.array(window, dtype=np.float32)

	def _getInfo(self) -> dict[str, Any]:
		step_idx = min(self.current_step, self.total_of_days - 1)
		return {
			"current_step": self.current_step,
			"total_cash": self.total_cash,
			"cash_portfolio": self.cash_portfolio,
			"percent_portfolio": self.percent_portfolio,
			"stock_portfolio": self.stock_portfolio,
			"current_closes": self.close_prices[step_idx],
			# "portfolio": {
			# 	f"{STOCKS[idx]}": f"{self.percent_portfolio[idx] * 100:.2f}% - R${self.cash_portfolio[idx]:.2f} - {self.stock_portfolio[idx]:.2f} cotas"
			# 	for idx in range(STOCKS_LEN)},
		}

	@staticmethod
	def _calculateReward(old_portfolio_value: float, new_portfolio_value: float) -> float:
		old_portfolio_value = max(old_portfolio_value, 1e-8)
		new_portfolio_value = max(new_portfolio_value, 1e-8)
		return float(np.log(new_portfolio_value / old_portfolio_value))

	# PORTFOLIO CALCULATE FUNCTIONS =================================
	@staticmethod
	def _calculateInitialCashPortfolio() -> list[float]:
		total_cents = round(INITIAL_CASH * 100)
		base_cents = total_cents // STOCKS_LEN
		remainder = total_cents % STOCKS_LEN
		return [(base_cents + (1 if i < remainder else 0)) / 100 for i in range(STOCKS_LEN)]

	def _calculateCashPortfolioByPercentPortfolio(self) -> list[float]:
		return [percent * self.total_cash for percent in self.percent_portfolio]

	def _calculateTotalCashByCashPortfolio(self) -> float:
		return sum(self.cash_portfolio)

	def _calculateCashPortfolioByStockPortfolio(self) -> list[float]:
		step_idx = min(self.current_step, self.total_of_days - 1)
		return [
			self.stock_portfolio[idx] * self.close_prices[step_idx][idx]
			for idx in range(STOCKS_LEN)
		]

	def _calculateStockPortfolioByCashPortfolio(self) -> list[float]:
		step_idx = min(self.current_step, self.total_of_days - 1)
		return [
			self.cash_portfolio[idx] / self.close_prices[step_idx][idx]
			for idx in range(STOCKS_LEN)
		]

	def _calculatePercentPortfolioByCashPortfolio(self) -> list[float]:
		self.total_cash = self._calculateTotalCashByCashPortfolio()
		if self.total_cash == 0:
			return [0.0] * STOCKS_LEN
		return [value / self.total_cash for value in self.cash_portfolio]