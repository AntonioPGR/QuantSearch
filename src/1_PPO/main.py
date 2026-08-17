import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PPOAgent import PPOAgent
from StocksEnv import StocksEnv


@dataclass
class Config:
    train_years: int = 3
    test_years: int = 1
    step_years: int = 1
    episodes: int = 8
    update_timestep: int = 256
    initial_capital: float = 10_000.0


class StandardScaler:
    """Small local scaler so PPO does not require scikit-learn."""
    def fit(self, values):
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, values):
        return (values - self.mean_) / self.scale_


def train_episode(env, agent, update_timestep=256):
    state, total = env.reset(), 0.0
    while True:
        _, reward, done, _ = env.step(agent.act(state))
        agent.memory.rewards.append(reward)
        agent.memory.is_terminals.append(done)
        total += reward
        if len(agent.memory.rewards) >= update_timestep or done:
            agent.update()
        if done:
            return total
        state = env._get_state()


def evaluate(env, agent):
    state, values = env.reset(), [env.portfolio_value]
    equal = [env.portfolio_value]
    equal_weights = np.full(env.n_stocks, 1.0 / env.n_stocks)
    while True:
        action = agent.policy_old.act(state)[1]
        state, _, done, info = env.step(action)
        equal.append(equal[-1] * float(np.dot(equal_weights, info["asset_returns"])))
        values.append(info["portfolio_value"])
        if done:
            break
    return {"start_value": values[0], "end_value": values[-1],
            "agent_return": values[-1] / values[0] - 1,
            "equal_weight_end": equal[-1], "equal_weight_return": equal[-1] / equal[0] - 1,
            "improvement_vs_equal_dollars": values[-1] - equal[-1]}


def main(args):
    cfg = Config(args.train_years, args.test_years, args.step_years, args.episodes,
                 args.update_timestep, args.initial_capital)
    base = StocksEnv(args.data_dir, initial_capital=cfg.initial_capital)
    dates = base.dates
    reports = []
    start = dates[0]
    while True:
        train_end = start + np.timedelta64(365 * cfg.train_years, "D")
        test_end = train_end + np.timedelta64(365 * cfg.test_years, "D")
        if test_end > dates[-1]:
            break
        train_raw = StocksEnv(args.data_dir, start_date=start, end_date=train_end,
                              initial_capital=cfg.initial_capital)
        scaler = StandardScaler().fit(train_raw.features.reshape(len(train_raw.dates), -1))
        train_env = StocksEnv(args.data_dir, start_date=start, end_date=train_end,
                              initial_capital=cfg.initial_capital, scaler=scaler)
        test_env = StocksEnv(args.data_dir, start_date=train_end, end_date=test_end,
                             initial_capital=cfg.initial_capital, scaler=scaler)
        agent = PPOAgent(train_env.state_dim, train_env.action_dim)
        for _ in range(cfg.episodes):
            train_episode(train_env, agent, cfg.update_timestep)
        report = evaluate(test_env, agent)
        report.update({"train_start": str(start.date()), "train_end": str(train_end.date()),
                       "test_start": str(train_end.date()), "test_end": str(test_end.date())})
        reports.append(report)
        agent.save(Path(args.output_dir) / f"ppo_{start.date()}.pt", train_env.feature_columns, train_env.stocks)
        agent.feature_importance(train_env.stocks, train_env.feature_columns)
        print(json.dumps(report, indent=2))
        start = start + np.timedelta64(365 * cfg.step_years, "D")
    if not reports:
        raise ValueError("Not enough common data for the configured train/test windows")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "walk_forward_report.json", "w") as f:
        json.dump(reports, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="_DATA/featured")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--update-timestep", type=int, default=256)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--output-dir", default="artifacts")
    main(parser.parse_args())
