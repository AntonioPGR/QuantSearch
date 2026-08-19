import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# LOCAL
from PPOMemory import PPOMemory


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def _distribution(self, state):
        x = F.tanh(self.actor_fc1(state))
        x = F.tanh(self.actor_fc2(x))
        return Normal(self.actor_out(x), torch.exp(self.action_logstd))

    def _value(self, state):
        x = F.tanh(self.critic_fc1(state))
        x = F.tanh(self.critic_fc2(x))
        return self.critic_out(x).squeeze(-1)

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        dist = self._distribution(state)
        raw = dist.sample()
        return raw.detach(), F.softmax(raw, dim=-1).detach().numpy(), dist.log_prob(raw).sum(), self._value(state).detach()

    def evaluate(self, states, actions):
        dist = self._distribution(states)
        return dist.log_prob(actions).sum(-1), self._value(states), dist.entropy().sum(-1)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10,
                 eps_clip=0.2, entropy_coef=0.01, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.entropy_coef = entropy_coef
        self.memory = PPOMemory()
        self.policy = ActorCritic(state_dim, action_dim)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_loss = nn.MSELoss()

    def act(self, state):
        with torch.no_grad():
            raw, weights, logprob, value = self.policy_old.act(state)
        self.memory.states.append(torch.as_tensor(state, dtype=torch.float32))
        self.memory.actions.append(raw)
        self.memory.logprobs.append(logprob)
        self.memory.state_values.append(value)
        return weights

    def update(self):
        if not self.memory.rewards:
            return
        returns, discounted = [], 0.0
        for reward, terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if terminal:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.tensor(returns, dtype=torch.float32)
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()
        old_values = torch.stack(self.memory.state_values).detach().reshape(-1)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = (-torch.min(ratios * advantages, clipped)
                    + 0.5 * self.value_loss(values, returns)
                    - self.entropy_coef * entropy).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def save(self, path, feature_columns=None, stocks=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "policy_old": self.policy_old.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "state_dim": self.policy.actor_fc1.in_features,
                    "action_dim": self.policy.actor_out.out_features, "stocks": stocks,
                    "feature_columns": feature_columns}, path)

    def feature_importance(self, stocks, feature_columns):
        """Global actor input importance: mean absolute first-layer weight."""
        weights = self.policy.actor_fc1.weight.detach().abs().mean(0).cpu().numpy()
        matrix = weights.reshape(len(stocks), len(feature_columns))
        result = {stock: {feature: float(matrix[i, j]) for j, feature in enumerate(feature_columns)}
                  for i, stock in enumerate(stocks)}
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/feature_importance.json", "w") as f:
            json.dump(result, f, indent=2)
        return result
