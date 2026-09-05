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
INITIAL_CASH = 10_000
DECIMAL_PLACES = 6
DATA_DIR = "data"


# ENV =================================
class NetworkTradingEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(self):
    super().__init__()
    # STEP
    self.current_step = OBS_LEN
    # DATA INFO
    self.close_prices = self._load_close_prices()  # now list[list[float]]
    self.total_of_days = len(self.close_prices)
    # PORTFOLIO INFO
    self.cash_portfolio = self._calculate_initial_cash_portfolio()
    self.percent_portfolio = self._calculate_percent_portfolio()
    self.stock_portfolio = self.calculate_stock_portfolio()
    self.current_value = self._calculate_portfolio_value()
    # ACTION INFO
    self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(OBS_LEN, STOCKS_LEN), dtype=np.float32)
    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(STOCKS_LEN,), dtype=np.float32)
    # ASSURE RESTART
    self.reset()
    self.render()

  # ENV FUNCTIONS =================================
  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
    super().reset(seed=SEED)
    # STEP
    self.current_step = OBS_LEN
    # PORTFOLIO INFO
    self.cash_portfolio = self._calculate_initial_cash_portfolio()
    self.percent_portfolio = self._calculate_percent_portfolio()
    # OBSERVATION INFO
    observation = self._get_observation()
    info = self._get_info()
    return observation, info

  def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    old_percent_portfolio = self.percent_portfolio
    self.percent_portfolio = action
    self.cash_portfolio = self.calculate_cash_portfolio()

    self.stock_portfolio = self.calculate_stock_portfolio()

    # self.portfolio = action
    new_portfolio_value = self._calculate_portfolio_value()
    reward = self._calculate_reward(old_portfolio_value, new_portfolio_value)
    self.current_step += 1
    terminated = self.current_step >= self.total_of_days
    truncated = False
    observation = self._get_observation()
    info = self._get_info()
    return observation, reward, terminated, truncated, info

  def render(self) -> None:
    items = self._get_info()
    for key, value in items.items():
      if isinstance(value, dict):
        for k, v in value.items():
          print(f"{k}: {v}")
      else:
        print(f"{key}: {value}")
      print("")

  # HELPER FUNCTIONS =================================
  @staticmethod
  def _load_close_prices() -> list[list[float]]:
    series_list = []
    lengths = []
    for stock in STOCKS:
      path = f"{DATA_DIR}/{stock}.csv"
      df = pd.read_csv(path, usecols=["Close"])
      close_series = pd.to_numeric(df["Close"], errors="coerce")
      close_series = close_series.dropna()
      if close_series.empty:
        raise ValueError(f"No valid Close prices found in {path}")
      # Convert to plain Python floats
      np_arr = close_series.to_numpy(dtype=np.float64)
      py_list = [float(x) for x in np_arr]
      series_list.append(py_list)
      lengths.append(len(py_list))
    if len(set(lengths)) != 1:
      raise ValueError(f"All stock files must contain the same number of valid Close prices. Lengths found: {dict(zip(STOCKS, lengths))}")
    days = lengths[0]
    prices = [[series_list[stock_idx][day_idx] for stock_idx in range(STOCKS_LEN)] for day_idx in range(days)]
    if any(p <= 0 for day in prices for p in day):
      raise ValueError("All closing prices must be greater than zero.")
    return prices

  @staticmethod
  def _calculate_initial_cash_portfolio() -> list[float]:
    total_cents = round(INITIAL_CASH * 100)
    base_cents = total_cents // STOCKS_LEN
    remainder = total_cents % STOCKS_LEN
    portfolio = [(base_cents + (1 if i < remainder else 0)) / 100 for i in range(STOCKS_LEN)]
    return portfolio

  def _calculate_cash_portfolio(self) -> list[float]:
    

  def calculate_stock_portfolio(self) -> list[float]:
    prices_prev = self.close_prices[self.current_step - 1]  # list[float]
    return [float(self.cash_portfolio[idx] / prices_prev[idx]) for idx in range(STOCKS_LEN)]

  def _calculate_percent_portfolio(self) -> list[float]:
    total_cash = self._calculate_portfolio_value()
    portfolio = [round(self.cash_portfolio[idx] / total_cash, 5) for idx in range(STOCKS_LEN)]
    return portfolio

  def _calculate_portfolio_value(self) -> float:
    total_value = 0.0
    for value in self.cash_portfolio:
      total_value += value
    return round(total_value, 2)

  def _get_observation(self) -> np.ndarray:
    # Keep observation as numpy for gym, but you could also return list if you want
    start = self.current_step - OBS_LEN
    end = self.current_step
    window = [self.close_prices[i][::] for i in range(start, end)]
    return np.array(window, dtype=np.float32)

  def _get_info(self) -> dict[str, Any]:
    current_closes = [float(x) for x in self.close_prices[self.current_step - 1]]
    return {"current_step": self.current_step, "total_cash": self._calculate_portfolio_value(), "cash_portfolio": self.cash_portfolio, "percent_portfolio": self.percent_portfolio, "stock_portfolio": self.stock_portfolio, "current_closes": current_closes, "portfolio": {f"{STOCKS[idx]}": f"{self.percent_portfolio[idx] * 100:.2f}% - R${self.cash_portfolio[idx]} - {self.stock_portfolio[idx]:.2f} cotas" for idx in range(STOCKS_LEN)}, }

  @staticmethod
  def _calculate_reward(old_portfolio_value: float, new_portfolio_value: float) -> float:
    old_portfolio_value = max(old_portfolio_value, 1e-8)
    new_portfolio_value = max(new_portfolio_value, 1e-8)
    return float(np.log(new_portfolio_value / old_portfolio_value))
