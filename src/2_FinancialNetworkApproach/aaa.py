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
DATA_DIR = "data"

# ENV =================================
class Teste(gym.Env):

  metadata = {"render_modes": []}

  def __init__(self):
    super().__init__()

    self.close_prices = self._load_close_prices()
    self.total_of_days = self.close_prices.shape[0]

    self.portfolio = self._calculate_initial_portfolio()
    self.cash = 0.0
    self.current_step = OBS_LEN

    self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(OBS_LEN, STOCKS_LEN), dtype=np.float32)
    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(STOCKS_LEN,), dtype=np.float32)

  # ENV FUNCTIONS =================================
  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,) -> tuple[ObsType, dict[str, Any]]:
    super().reset(seed=seed)

    self.current_step = OBS_LEN
    self.portfolio = self._calculate_initial_portfolio()
    self.cash = 0.0

    observation = self._get_observation()
    info = self._get_info()

    return observation, info

  def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    action = np.asarray(action, dtype=np.float32)
    action = self._normalize_action(action)

    old_portfolio_value = self._calculate_portfolio_value()

    self.portfolio = self._calculate_new_portfolio(action)

    self.current_step += 1

    terminated = self.current_step >= self.total_of_days
    truncated = False

    new_portfolio_value = self._calculate_portfolio_value()
    reward = self._calculate_reward(old_portfolio_value, new_portfolio_value)

    observation = self._get_observation()
    info = self._get_info()

    return observation, reward, terminated, truncated, info

  def render(self) -> RenderFrame | list[RenderFrame] | None:
    print(f"Day: {self.current_step}")
    print(f"Portfolio value: {self._calculate_portfolio_value()}")
    print("Portfolio:")

    for idx in range(STOCKS_LEN):
      print(f"{STOCKS[idx]}: {self.portfolio[idx]}")

    print(f"Cash: {self.cash}")

    return None

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

  def _calculate_initial_portfolio(self) -> np.ndarray:
    """
    Creates an equal-value portfolio at the first trading day.

    The portfolio contains the number of shares held for each stock.
    Each stock receives the same monetary value.
    """

    initial_cash = 10_000.0
    first_prices = self.close_prices[OBS_LEN - 1]
    amount_per_stock = initial_cash / STOCKS_LEN

    portfolio = amount_per_stock / first_prices

    return portfolio.astype(np.float32)

  def _calculate_portfolio_value(self) -> float:
    """
    Returns the total value of all stock positions and cash
    at the current step.
    """

    current_prices = self.close_prices[min(self.current_step, self.total_of_days - 1)]
    portfolio_value = np.sum(self.portfolio * current_prices)

    return float(portfolio_value + self.cash)

  def _calculate_new_portfolio(self, action: np.ndarray) -> np.ndarray:
    """
    Converts the action into portfolio percentages and calculates
    the number of shares for each stock.
    """

    current_prices = self.close_prices[min(self.current_step, self.total_of_days - 1)]
    current_portfolio_value = self._calculate_portfolio_value()

    target_values = action * current_portfolio_value
    new_portfolio = target_values / current_prices

    return new_portfolio.astype(np.float32)

  def _normalize_action(self, action: np.ndarray) -> np.ndarray:
    """
    Converts the agent action into percentages whose sum is 1.
    """

    if action.shape != (STOCKS_LEN,):
      raise ValueError(f"Action must have shape {(STOCKS_LEN,)}, but received {action.shape}")

    action = np.clip(action, 0.0, 1.0)
    action_sum = np.sum(action)

    if action_sum == 0:
      return np.ones(STOCKS_LEN, dtype=np.float32) / STOCKS_LEN

    return action / action_sum

  def _get_observation(self) -> np.ndarray:
    start = self.current_step - OBS_LEN
    end = self.current_step

    window = self.close_prices[start:end].copy()
    last_prices = window[-1, :].copy()
    window = window / last_prices

    return window.astype(np.float32)

  def _get_info(self) -> dict[str, Any]:
    return {
      "current_step": self.current_step,
      "cash": self.cash,
      "portfolio_value": self._calculate_portfolio_value(),
      "portfolio": {f"{stock}": value for stock, value in zip(STOCKS, self.portfolio)},
    }

  def _calculate_reward(self, old_portfolio_value: float, new_portfolio_value: float) -> float:
    old_portfolio_value = max(old_portfolio_value, 1e-8)
    new_portfolio_value = max(new_portfolio_value, 1e-8)

    return float(np.log(new_portfolio_value / old_portfolio_value))