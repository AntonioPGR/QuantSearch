
#pragma once
#include <torch/torch.h>
#include <tuple>
 
// ---------------------------------------------------------------------------
// Observation bundle. Every field carries a leading batch dimension.
//   x3d_price / x3d_volume : [B, n_stocks*n_link_features, D, H, W]
//   x2d_price / x2d_volume : [B, n_stocks*n_node_features, H, W]
//   x_portfolio            : [B, n_stocks]   (current weights)
// ---------------------------------------------------------------------------
struct Observation {
  torch::Tensor x3d_price;
  torch::Tensor x3d_volume;
  torch::Tensor x2d_price;
  torch::Tensor x2d_volume;
  torch::Tensor x_portfolio;
 
  Observation to(torch::Device d) const {
    return {x3d_price.to(d), x3d_volume.to(d), x2d_price.to(d),
            x2d_volume.to(d), x_portfolio.to(d)};
  }
 
  // Select a minibatch along the batch dimension.
  Observation index(const torch::Tensor& idx) const {
    return {x3d_price.index_select(0, idx), x3d_volume.index_select(0, idx),
            x2d_price.index_select(0, idx), x2d_volume.index_select(0, idx),
            x_portfolio.index_select(0, idx)};
  }
 
  static Observation stack(const std::vector<Observation>& v) {
    std::vector<torch::Tensor> a, b, c, d, e;
    a.reserve(v.size()); b.reserve(v.size()); c.reserve(v.size());
    d.reserve(v.size()); e.reserve(v.size());
    for (const auto& o : v) {
      a.push_back(o.x3d_price);  b.push_back(o.x3d_volume);
      c.push_back(o.x2d_price);  d.push_back(o.x2d_volume);
      e.push_back(o.x_portfolio);
    }
    return {torch::cat(a, 0), torch::cat(b, 0), torch::cat(c, 0),
            torch::cat(d, 0), torch::cat(e, 0)};
  }
};
 
// What act() hands back to the rollout loop.
struct ActionOutput {
  torch::Tensor action;    // [B] int64
  torch::Tensor log_prob;  // [B]
  torch::Tensor value;     // [B]
};
 
struct AgentImpl : torch::nn::Module {
  // ---- fixed feature sizes, derived from the adaptive pools below ----------
  static constexpr int64_t kConvChannels = 64;
  static constexpr int64_t kPool3d       = 2;  // AdaptiveAvgPool3d({2,2,2})
  static constexpr int64_t kPool2d       = 3;  // AdaptiveAvgPool2d({3,3})
  static constexpr int64_t kFeat3d = kConvChannels * kPool3d * kPool3d * kPool3d; // 512
  static constexpr int64_t kFeat2d = kConvChannels * kPool2d * kPool2d;           // 576
  static constexpr int64_t kFeatVec = 64;
 
  int64_t n_stocks_;
 
  torch::nn::Sequential conv3d_price{nullptr};
  torch::nn::Sequential conv3d_volume{nullptr};
  torch::nn::Sequential conv2d_price{nullptr};
  torch::nn::Sequential conv2d_volume{nullptr};
  torch::nn::Sequential mlp_portfolio{nullptr};
  torch::nn::Sequential mlp_shared{nullptr};
  torch::nn::Linear actor{nullptr};
  torch::nn::Linear critic{nullptr};
 
  AgentImpl(int64_t n_stocks, int64_t n_node_features, int64_t n_link_features)
      : n_stocks_(n_stocks) {
    using namespace torch::nn;
 
    const int64_t c3 = n_stocks * n_link_features;  // channels of the 3D inputs
    const int64_t c2 = n_stocks * n_node_features;  // channels of the 2D inputs
 
    auto make3d = [&] {
      return Sequential(
          Conv3d(Conv3dOptions(c3, 32, 3).padding(1)), ReLU(),
          Conv3d(Conv3dOptions(32, kConvChannels, 3).padding(1)), ReLU(),
          AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions({kPool3d, kPool3d, kPool3d})));
    };
    auto make2d = [&] {
      return Sequential(
          Conv2d(Conv2dOptions(c2, 32, 3).padding(1)), ReLU(),
          Conv2d(Conv2dOptions(32, kConvChannels, 3).padding(1)), ReLU(),
          AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({kPool2d, kPool2d})));
    };
 
    conv3d_price  = register_module("conv3d_price",  make3d());
    conv3d_volume = register_module("conv3d_volume", make3d());
    conv2d_price  = register_module("conv2d_price",  make2d());
    conv2d_volume = register_module("conv2d_volume", make2d());
 
    mlp_portfolio = register_module(
        "mlp_portfolio",
        Sequential(Linear(n_stocks, kFeatVec), ReLU(),
                   Linear(kFeatVec, kFeatVec), ReLU()));
 
    const int64_t total_features = 2 * kFeat3d + 2 * kFeat2d + kFeatVec;  // 2240
 
    mlp_shared = register_module(
        "mlp_shared",
        Sequential(Linear(total_features, 512), ReLU(), Linear(512, 256), ReLU()));
 
    actor  = register_module("actor",  Linear(256, n_stocks));
    critic = register_module("critic", Linear(256, 1));
 
    // Small actor init keeps the initial policy close to uniform, which makes
    // early PPO updates much better behaved.
    torch::NoGradGuard ng;
    actor->weight.mul_(0.01);
    actor->bias.zero_();
  }
 
  // Returns {action_logits [B, n_stocks], value [B]}.
  std::tuple<torch::Tensor, torch::Tensor> forward(const Observation& o) {
    auto f3a = conv3d_price->forward(o.x3d_price).flatten(1);
    auto f3b = conv3d_volume->forward(o.x3d_volume).flatten(1);
    auto f2a = conv2d_price->forward(o.x2d_price).flatten(1);
    auto f2b = conv2d_volume->forward(o.x2d_volume).flatten(1);
    auto fv  = mlp_portfolio->forward(o.x_portfolio);
 
    auto features = torch::cat({f3a, f3b, f2a, f2b, fv}, 1);
    features = mlp_shared->forward(features);
 
    auto logits = actor->forward(features);
    auto value  = critic->forward(features).squeeze(-1);
    return {logits, value};
  }
 
  // Rollout-time: sample an action, no gradients.
  ActionOutput act(const Observation& o, bool deterministic = false) {
    torch::NoGradGuard ng;
    auto [logits, value] = forward(o);
    auto log_probs = torch::log_softmax(logits, -1);
 
    torch::Tensor action;
    if (deterministic) {
      action = std::get<1>(log_probs.max(-1));
    } else {
      action = torch::multinomial(log_probs.exp(), 1).squeeze(-1);
    }
    auto lp = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1);
    return {action, lp, value};
  }
 
  // Update-time: re-score stored actions, gradients on.
  // Returns {log_prob [B], entropy [B], value [B]}.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  evaluate_actions(const Observation& o, const torch::Tensor& actions) {
    auto [logits, value] = forward(o);
    auto log_probs = torch::log_softmax(logits, -1);
    auto lp = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1);
    auto entropy = -(log_probs.exp() * log_probs).sum(-1);
    return {lp, entropy, value};
  }
 
  // Convenience: turn a discrete action into portfolio weights (one-hot).
  torch::Tensor action_to_weights(const torch::Tensor& action) const {
    return torch::one_hot(action, n_stocks_).to(torch::kFloat32);
  }
};
TORCH_MODULE(Agent);  // defines the holder class `Agent` for `AgentImpl`
 
// ---------------------------------------------------------------------------
// Continuous variant (if you want real weights instead of "pick one stock"):
//
//   torch::Tensor log_std = register_parameter("log_std", torch::zeros({n_stocks}));
//
//   act():      auto std = log_std.exp();
//               auto raw = logits + torch::randn_like(logits) * std;
//               auto lp  = (-0.5 * ((raw - logits) / std).pow(2)
//                           - log_std - 0.5 * std::log(2 * M_PI)).sum(-1);
//               weights  = torch::softmax(raw, -1);
//   entropy:    (0.5 + 0.5 * std::log(2 * M_PI) + log_std).sum(-1);
//
// Everything downstream (GAE, PPO loss) is identical; only the distribution
// changes, and `action` becomes a [B, n_stocks] float tensor.
// ---------------------------------------------------------------------------
 