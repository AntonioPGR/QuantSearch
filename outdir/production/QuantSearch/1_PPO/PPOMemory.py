import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor Network
        self.actor_fc1 = nn.Linear(state_dim, 128)
        self.actor_fc2 = nn.Linear(128, 128)
        self.actor_out = nn.Linear(128, action_dim)
        # Learnable standard deviation for continuous exploration
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        # Critic Network
        self.critic_fc1 = nn.Linear(state_dim, 128)
        self.critic_fc2 = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        # Forward pass through Actor
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_mean = self.actor_out(x)
        action_std = torch.exp(self.action_logstd)
        # Sample raw action from a Normal distribution
        dist = Normal(action_mean, action_std)
        raw_action = dist.sample()
        action_logprob = dist.log_prob(raw_action).sum()
        # Forward pass through Critic
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(x_c))
        state_value = self.critic_out(x_c)
        # Convert raw action to portfolio percentages summing to 1.0
        portfolio_weights = F.softmax(raw_action, dim=-1)
        return raw_action.detach(), portfolio_weights.detach().numpy(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_mean = self.actor_out(x)
        action_std = torch.exp(self.action_logstd)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(x_c))
        state_values = self.critic_out(x_c)
        return action_logprobs, state_values.squeeze(), dist_entropy