from typing import SupportsFloat, Any
import gymnasium as gym
import pandas as pd
from gymnasium.core import ActType, ObsType, RenderFrame
import numpy as np
from gymnasium import spaces

# SETTINGS ============================
SEED = 42
STOCKS = ["SBSP3", "BBDC3", "BRAP4", "VALE3", "GGBR4", "ISAE4", "CSNA3", "CMIG4", "EMBJ3", "CPLE3", "USIM5", "ITSA4", "AXIA3", "VIVT3", "WEGE3", "POMO4", "PETR4", "PETR3"]
STOCKS_LEN = len(STOCKS)
OBS_LEN = 50
INITIAL_CASH = 10_0000
DECIMAL_PLACES = 6
DATA_DIR = "data"

# ENV =================================
class NetworkTradingEnv(gym.Env):
	
	metadata = {"render_modes": []}
	
	def __init__(self):
		super().__init__()
		# DATA INFO
		self.close_prices = self._load_close_prices()
		self.total_of_days = self.close_prices.shape[0]
		# PORTFOLIO INFO
		self.cash_portfolio = self._calculate_initial_cash_portfolio()
		self.percent_portfolio = self._calculate_percent_portfolio()
		self.cash = self._calculate_portfolio_value()
		# ACTION INFO
		self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(OBS_LEN, STOCKS_LEN), dtype=np.float32)
		self.action_space = spaces.Box(low=0.0, high=1.0, shape=(STOCKS_LEN,), dtype=np.float32)
		# STEP
		self.current_step = OBS_LEN
		# ASSURE RESTART
		self.reset()
	
	# ENV FUNCTIONS =================================
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,) -> tuple[ObsType, dict[str, Any]]:
		super().reset(seed=SEED)
		# PORTFOLIO INFO
		self.cash_portfolio = self._calculate_initial_cash_portfolio()
		self.percent_portfolio = self._calculate_percent_portfolio()
		self.cash = self._calculate_portfolio_value()
		# STEP
		self.current_step = OBS_LEN
		# OBSERVATION INFO
		observation = self._get_observation()
		info = self._get_info()
		return observation, info
	
	def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		old_portfolio_value = self._calculate_portfolio_value()
		# self.portfolio = action
		new_portfolio_value = self._calculate_portfolio_value()
		reward = self._calculate_reward(old_portfolio_value, new_portfolio_value)
		
		self.current_step += 1
		terminated = self.current_step >= self.total_of_days
		truncated = False
		observation = self._get_observation()
		info = self._get_info()
		return observation, reward, terminated, truncated, info
	
	def close(self) -> None:
		return None
	
	# HELPER FUNCTIONS =================================
	def _load_close_prices(self) -> np.ndarray:
		series_list = []
		lengths = []
		for stock in STOCKS:
			path = f"{DATA_DIR}/{stock}.csv"
			df = pd.read_csv(path, usecols=["Close"])
			close_series = pd.to_numeric(df["Close"], errors="coerce")
			close_series = close_series.dropna()
			if close_series.empty:
				raise ValueError(f"No valid Close prices found in {path}")
			series_list.append(close_series.to_numpy(dtype=np.float32))
			lengths.append(len(close_series))
		if len(set(lengths)) != 1:
			raise ValueError(f"All stock files must contain the same number of valid Close prices. Lengths found: {dict(zip(STOCKS, lengths))}")
		prices = np.column_stack(series_list)
		if np.any(prices <= 0):
			raise ValueError("All closing prices must be greater than zero.")
		return prices.astype(np.float32)
	
	def _calculate_initial_cash_portfolio(self) -> list[float]:
		return [INITIAL_CASH/STOCKS_LEN for _ in range(STOCKS_LEN)]
	
	def _calculate_percent_portfolio(self):
		portfolio = []
		for idx in range(STOCKS_LEN):
			portfolio.append(self.cash_portfolio[idx] / self.close_prices[self.current_step - 1, idx])
		return portfolio
	
	def _calculate_portfolio_value(self) -> float:
		total_value = 0.0
		for value in self.cash_portfolio:
			total_value += value
		return total_value
	
	def _get_observation(self) -> np.ndarray:
		start = self.current_step - OBS_LEN
		end = self.current_step
		window = self.close_prices[start:end].copy()
		return window
	
	def _get_info(self) -> dict[str, Any]:
		return {
			"current_step": self.current_step,
			"cash": self._calculate_portfolio_value(),
			"portfolio": {
				f"{stock}": value for stock, value in zip(STOCKS, self.portfolio)
			},
		}
	
	def _calculate_reward(self, old_portfolio_value: float, new_portfolio_value: float) -> float:
		old_portfolio_value = max(old_portfolio_value, 1e-8)
		new_portfolio_value = max(new_portfolio_value, 1e-8)
		return np.log(new_portfolio_value / old_portfolio_value)