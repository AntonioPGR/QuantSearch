#include <torch/torch.h>
#include <vector>
#include "agent.cpp"

using namespace std;
using namespace torch;
using namespace torch::nn::utils;

struct PPOConfig {
  int64_t rollout_steps  = 2048;
  int64_t epochs         = 5;
  int64_t minibatch_size = 64;
  double  gamma          = 0.99;
  double  lambda         = 0.95;
  double  clip_range     = 0.2;
  double  value_coef     = 0.5;
  double  entropy_coef   = 0.01;
  double  max_grad_norm  = 0.5;
  double  learning_rate  = 3e-4;
  double  weight_decay   = 1e-4;
};

class RolloutBuffer {
 public:
  void add(const Observation& obs, const Tensor& action, const Tensor& log_prob, const Tensor& value, float reward, bool done) {
    obs_.push_back(obs);
    actions_.push_back(action);
    log_probs_.push_back(log_prob);
    values_.push_back(value);
    rewards_.push_back(reward);
    dones_.push_back(done ? 1.0f : 0.0f);
  }

  void clear() {
    obs_.clear(); actions_.clear(); log_probs_.clear();
    values_.clear(); rewards_.clear(); dones_.clear();
  }

  int64_t size() const { return static_cast<int64_t>(rewards_.size()); }

  // Generalised Advantage Estimation, walked backwards through the rollout.
  // `last_value` is V(s_T) for bootstrapping a truncated episode.
  void compute_gae(double gamma, double lambda, float last_value, bool last_done) {
    const int64_t T = size();
    advantages_ = torch::zeros({T});
    float gae = 0.0f;

    for (int64_t t = T - 1; t >= 0; --t) {
      float next_value    = (t == T - 1) ? last_value : values_[t + 1].item<float>();
      float next_non_term = (t == T - 1) ? (last_done ? 0.0f : 1.0f) : (1.0f - dones_[t]);
      float v     = values_[t].item<float>();
      float delta = rewards_[t] + static_cast<float>(gamma) * next_value * next_non_term - v;
      gae = delta + static_cast<float>(gamma * lambda) * next_non_term * gae;
      advantages_[t] = gae;
    }
    returns_ = advantages_ + flat_values();
  }

  Observation obs() const { return Observation::stack(obs_); }
  Tensor actions()    const { return cat(actions_, 0); }
  Tensor log_probs()  const { return cat(log_probs_, 0); }
  Tensor flat_values()const { return cat(values_, 0); }
  Tensor advantages() const { return advantages_; }
  Tensor returns()    const { return returns_; }

 private:
  vector<Observation> obs_;
  vector<Tensor> actions_, log_probs_, values_;
  vector<float>         rewards_, dones_;
  Tensor              advantages_, returns_;
};

struct PPOStats {
  float policy_loss = 0, value_loss = 0, entropy = 0, approx_kl = 0;
};

// One full PPO update: several epochs of shuffled minibatches over the rollout.
inline PPOStats ppo_update(Agent& agent, optim::Optimizer& optimizer,
                           const RolloutBuffer& buf, const PPOConfig& cfg) {
  if (buf.size() == 0) return {};

  auto obs          = buf.obs();
  auto actions      = buf.actions();
  auto old_log_prob = buf.log_probs().detach();
  auto returns      = buf.returns().detach();
  auto advantages   = buf.advantages().detach();

  // Normalising advantages is what keeps the clipped objective well scaled.
  auto advantage_std = advantages.numel() > 1 ? advantages.std() : torch::ones({}, advantages.options());
  advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8);

  const int64_t N = buf.size();
  PPOStats stats;
  int64_t n_updates = 0;

  for (int64_t epoch = 0; epoch < cfg.epochs; ++epoch) {
    auto perm = torch::randperm(N, kInt64);

    for (int64_t start = 0; start < N; start += cfg.minibatch_size) {
      auto idx = perm.slice(0, start, min(start + cfg.minibatch_size, N));

      auto mb_obs = obs.index(idx);
      auto mb_act = actions.index_select(0, idx);
      auto mb_old = old_log_prob.index_select(0, idx);
      auto mb_adv = advantages.index_select(0, idx);
      auto mb_ret = returns.index_select(0, idx);

      auto [log_prob, entropy, value] = agent->evaluate_actions(mb_obs, mb_act);

      // Clipped surrogate objective.
      auto ratio = (log_prob - mb_old).exp();
      auto surr1 = ratio * mb_adv;
      auto surr2 = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv;
      auto policy_loss = -min(surr1, surr2).mean();

      auto value_loss   = mse_loss(value, mb_ret);
      auto entropy_loss = -entropy.mean();

      auto loss = policy_loss
                + cfg.value_coef   * value_loss
                + cfg.entropy_coef * entropy_loss;

      optimizer.zero_grad();
      loss.backward();
      clip_grad_norm_(agent->parameters(), cfg.max_grad_norm);
      optimizer.step();

      {
        NoGradGuard ng;
        stats.policy_loss += policy_loss.item<float>();
        stats.value_loss  += value_loss.item<float>();
        stats.entropy     += entropy.mean().item<float>();
        stats.approx_kl   += (mb_old - log_prob).mean().item<float>();
      }
      ++n_updates;
    }
  }

  if (n_updates > 0) {
    stats.policy_loss /= n_updates;
    stats.value_loss  /= n_updates;
    stats.entropy     /= n_updates;
    stats.approx_kl   /= n_updates;
  }
  return stats;
}