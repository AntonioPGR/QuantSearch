"""Feature-only portfolio environment.

At date t the agent sees features known at t and chooses the portfolio held
from t to t+1.  The reward is log(V[t+1] / V[t]).  OHLCV is never included in
the observation; Close is used only to calculate the realised reward.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class StocksEnv:
    DEFAULT_STOCKS = [
        "VALE3", "ITUB4", "PETR4", "AXIA3", "PETR3",
        "BBDC4", "ITSA4", "SBSP3", "B3SA3", "WEGE3",
    ]
    PRICE_COLUMNS = {"Close", "High", "Low", "Open", "Volume"}

    def __init__(self, data_dir="_DATA/featured", stocks=None, start_date=None,
                 end_date=None, initial_capital=10_000.0, scaler=None):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute() and not self.data_dir.exists():
            project_src = Path(__file__).resolve().parents[1]
            candidate = project_src / self.data_dir
            if candidate.exists():
                self.data_dir = candidate
        self.stocks = list(stocks or self.DEFAULT_STOCKS)
        self.initial_capital = float(initial_capital)
        self.scaler = scaler

        frames = {}
        for stock in self.stocks:
            path = self.data_dir / f"{stock}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing data for {stock}: {path}")
            df = pd.read_csv(path)
            if "Date" not in df or "Close" not in df:
                raise ValueError(f"{path} must contain Date and Close columns")
            df["Date"] = pd.to_datetime(df["Date"])
            feature_columns = [c for c in df.columns if c not in self.PRICE_COLUMNS and c != "Date"]
            if not feature_columns:
                raise ValueError(
                    f"{path} has no features. Run the feature-generation step or use _DATA/featured."
                )
            frames[stock] = (df.set_index("Date").sort_index(), feature_columns)

        common_dates = None
        for df, _ in frames.values():
            common_dates = df.index if common_dates is None else common_dates.intersection(df.index)
        dates = pd.DatetimeIndex(common_dates).sort_values()
        if start_date is not None:
            dates = dates[dates >= pd.Timestamp(start_date)]
        if end_date is not None:
            dates = dates[dates <= pd.Timestamp(end_date)]
        if len(dates) < 2:
            raise ValueError("The selected date range must contain at least two common dates")

        # All generated files use the same feature set. Taking the intersection
        # makes this robust to a file with one extra generated indicator.
        feature_columns = list(frames[self.stocks[0]][1])
        for stock in self.stocks[1:]:
            feature_columns = [c for c in feature_columns if c in frames[stock][0].columns]
        self.feature_columns = feature_columns
        self.dates = dates
        self.features = np.stack([
            frames[stock][0].loc[dates, feature_columns].to_numpy(dtype=np.float32)
            for stock in self.stocks
        ], axis=1)
        self.close_prices = np.stack([
            frames[stock][0].loc[dates, "Close"].to_numpy(dtype=np.float64)
            for stock in self.stocks
        ], axis=1)

        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        if scaler is not None:
            self.features = scaler.transform(self.features.reshape(len(dates), -1)).reshape(self.features.shape)
        self.n_stocks = len(self.stocks)
        self.n_features = len(self.feature_columns)
        self.state_dim = self.n_stocks * self.n_features
        self.action_dim = self.n_stocks
        self.max_steps = len(self.dates) - 1
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cur_portfolio = np.full(self.n_stocks, 1.0 / self.n_stocks, dtype=np.float64)
        self.portfolio_value = self.initial_capital
        return self._get_state()

    def _get_state(self):
        return self.features[self.current_step].reshape(-1).astype(np.float32)

    def step(self, action):
        weights = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(weights) != self.n_stocks or not np.all(np.isfinite(weights)):
            raise ValueError("Action must contain one finite weight per stock")
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum() if weights.sum() > 0 else 1.0

        asset_returns = self.close_prices[self.current_step + 1] / self.close_prices[self.current_step]
        gross_return = float(np.dot(weights, asset_returns))
        reward = float(np.log(max(gross_return, 1e-12)))
        self.portfolio_value *= gross_return
        self.cur_portfolio = weights
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return (np.zeros(self.state_dim, dtype=np.float32) if done else self._get_state()), reward, done, {
            "date": self.dates[self.current_step],
            "portfolio_value": self.portfolio_value,
            "weights": weights.copy(),
            "asset_returns": asset_returns,
        }

    @property
    def current_date(self):
        return self.dates[self.current_step]
