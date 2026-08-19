from pathlib import Path
import numpy as np
import pandas as pd
# LOCAL
from Config import Config


class StocksEnv:

    data_dir = Path(Config.DATA_DIR)
    stocks = Config.STOCKS
    price_cols = Config.PRICE_COLUMNS

    feature_cols = None
    closes_prices = {}
    dates = None
    features = None

    n_stocks = len(Config.STOCKS)
    n_features = None
    state_dim = None
    action_dim = None

    cur_step = 0
    portfolio = []

    def __init__(self):
        self.feature_cols = [col for col in pd.read_csv(self.data_dir / f"{Config.STOCKS[0]}.csv").columns if col not in self.price_cols and col != "Date"]
        self.n_features = len(self.feature_cols)

        dfs = {}
        for stock in Config.STOCKS:
            path = self.data_dir / f"{stock}.csv"
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            dfs[stock] = (df.set_index("Date").sort_index(), self.feature_cols)
            self.closes_prices[stock] = dfs[stock]["Close"]
        self.dates = pd.DatetimeIndex(dfs[0].index).sort_values()

        self.state_dim = self.n_stocks * self.n_features
        self.action_dim = self.n_stocks
        self.reset()

    def reset(self):
        self.cur_step = 0
        self.portfolio = np.full(self.n_stocks, 1.0 / self.n_stocks, dtype=np.float64)
        return self._get_state()

    def _get_state(self):
        return self.features[self.cur_step].reshape(-1).astype(np.float32)

    def step(self, action):
        weights = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(weights) != self.n_stocks or not np.all(np.isfinite(weights)):
            raise ValueError("Action must contain one finite weight per stock")
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum() if weights.sum() > 0 else 1.0

        asset_returns = self.close_prices[self.cur_step + 1] / self.close_prices[self.cur_step]
        gross_return = float(np.dot(weights, asset_returns))
        reward = float(np.log(max(gross_return, 1e-12)))
        self.portfolio_value *= gross_return
        self.cur_portfolio = weights
        self.cur_step += 1
        done = self.cur_step >= self.max_steps
        return (np.zeros(self.state_dim, dtype=np.float32) if done else self._get_state()), reward, done, {
            "date": self.dates[self.cur_step],
            "portfolio_value": self.portfolio_value,
            "weights": weights.copy(),
            "asset_returns": asset_returns,
        }


    # GETTERS =======================================
    def getCurrentDate(self):
        return self.dates[self.cur_step]

    def getCurrentPrice(self, stock):
        return self.closes_prices[stock][self.cur_step]

    def getPortfolioValue(self):
        value = 0
        for idx in range(self.n_stocks):
            value += self.portfolio[idx] * self.getCurrentPrice(self.stocks[idx])
        return value
