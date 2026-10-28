#pragma once

#include <torch/torch.h>
#include <string>
#include <tuple>
#include <vector>

#include "config.cpp"
#include "observation.cpp"

using namespace std;
using namespace torch;
using namespace torch::nn;

// CONSTANTS USED ==================================================================================
// B  = Batch size (1 when acting, minibatch size when PPO is updating)
// N  = Number of stocks (amount of assets in the portfolio)
// F  = Amount of node-level features      (NetworkFeatureExtractor::nodeLevelFeaturesAmount)
// L  = Amount of link-level features      (NetworkFeatureExtractor::linkLevelFeaturesAmount)
// K1 = Outputs of each 1D convolution     (kOut1d = 2 values per node feature)
// K2 = Outputs of each 2D convolution     (kOut2d = 4 values per link feature)
// C1 = Hidden channels inside each 1D conv (kHidden1d)
// C2 = Hidden channels inside each 2D conv (kHidden2d)
// G  = Number of networks (price network + volume network = 2)
// S  = Size of the PPO input vector = G * (F * K1 + L * K2) + N
//
// FLOW ================================================================================================================
// Price network  -> F node columns [B, 1, N] -> F x Conv1d -> F x [B, K1] -\
//                -> L link matrix  [B, 1, N, N] -> L x Conv2d -> L x [B, K2] --\
// Volume network -> F node columns [B, 1, N] -> F x Conv1d -> F x [B, K1] ----> cat -> [B, S] -> mlp_shared -> actor  [B, N]
//                -> L link matrix  [B, 1, N, N] -> L x Conv2d -> L x [B, K2] --/                             -> critic [B]
// Portfolio percents [B, N] ------------------------------------------------/
//
// Every Conv1d / Conv2d is an independent module (own weights), registered in the agent, so its parameters are inside
// agent->parameters() and are trained by the same optimizer and the same PPO loss as the actor and the critic.

struct ActionOutput {
  Tensor action;    // [B, N] sampled allocation scores (before softmax), stored for PPO to re-score
  Tensor log_prob;  // [B]
  Tensor value;     // [B]
  vector<double> weights; // [N] in percent of how much money to allocate to each stock (sums to 1)
};

struct AgentImpl : Module {
  static constexpr int64_t kHidden1d = 8;  // C1
  static constexpr int64_t kOut1d    = 2;  // K1
  static constexpr int64_t kHidden2d = 8;  // C2
  static constexpr int64_t kOut2d    = 4;  // K2
  static constexpr int64_t kNetworks = 2;  // G

  int64_t n_stocks_;
  int64_t n_node_features_;
  int64_t n_link_features_;
  Tensor log_std;

  vector<Sequential> conv1d_price;   // F modules, conv1d_price[f] sees only node feature f of the price network
  vector<Sequential> conv1d_volume;  // F modules, conv1d_volume[f] sees only node feature f of the volume network
  vector<Sequential> conv2d_price;   // L modules, conv2d_price[l] sees only link feature l of the price network
  vector<Sequential> conv2d_volume;  // L modules, conv2d_volume[l] sees only link feature l of the volume network
  Sequential mlp_shared{nullptr};
  Linear actor{nullptr};
  Linear critic{nullptr};

  AgentImpl(int64_t n_stocks, int64_t n_node_features, int64_t n_link_features): n_stocks_(n_stocks), n_node_features_(n_node_features), n_link_features_(n_link_features) {

    // [B, 1, N] -> Conv1d -> [B, C1, N] -> ReLU -> Conv1d -> [B, K1, N] -> AvgPool over the N stocks -> [B, K1, 1]
    auto make1d = [&] {
      return Sequential(
        Conv1d(Conv1dOptions(1, kHidden1d, 3).padding(1)),
        ReLU(),
        Conv1d(Conv1dOptions(kHidden1d, kOut1d, 3).padding(1)),
        AdaptiveAvgPool1d(AdaptiveAvgPool1dOptions(1))
      );
    };
    // [B, 1, N, N] -> Conv2d -> [B, C2, N, N] -> ReLU -> Conv2d -> [B, K2, N, N] -> AvgPool over the N x N links -> [B, K2, 1, 1]
    auto make2d = [&] {
      return Sequential(
        Conv2d(Conv2dOptions(1, kHidden2d, 3).padding(1)),
        ReLU(),
        Conv2d(Conv2dOptions(kHidden2d, kOut2d, 3).padding(1)),
        AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({1, 1}))
      );
    };

    for (int64_t f = 0; f < n_node_features; ++f) {
      conv1d_price.push_back(register_module("conv1d_price_" + to_string(f), make1d()));
      conv1d_volume.push_back(register_module("conv1d_volume_" + to_string(f), make1d()));
    }
    for (int64_t l = 0; l < n_link_features; ++l) {
      conv2d_price.push_back(register_module("conv2d_price_" + to_string(l), make2d()));
      conv2d_volume.push_back(register_module("conv2d_volume_" + to_string(l), make2d()));
    }

    // S = G * (F * K1 + L * K2) + N  ->  2 * (8 * 2 + 8 * 4) + N = 96 + N
    int64_t total_features = kNetworks * (n_node_features * kOut1d + n_link_features * kOut2d) + n_stocks;

    mlp_shared = register_module("mlp_shared", Sequential(Linear(total_features, 512), ReLU(), Linear(512, 256), ReLU()));

    actor  = register_module("actor",  Linear(256, n_stocks));
    critic = register_module("critic", Linear(256, 1));
    log_std = register_parameter("log_std", torch::full({n_stocks}, -0.5));

    NoGradGuard ng;
    actor->weight.mul_(0.01);
    actor->bias.zero_();
  }

  // Returns {allocation means [B, N], value [B]}.
  tuple<Tensor, Tensor> forward(const Observation& o) {
    vector<Tensor> features;  // will hold 2 * F tensors of [B, K1], 2 * L tensors of [B, K2] and 1 tensor of [B, N]

    // 1D: o.x1d_*.slice(1, f, f + 1) = column f of the node features matrix -> [B, 1, N] -> [B, K1]
    for (int64_t f = 0; f < n_node_features_; ++f) {
      features.push_back(conv1d_price[f]->forward(o.x1d_price.slice(1, f, f + 1)).flatten(1));
      features.push_back(conv1d_volume[f]->forward(o.x1d_volume.slice(1, f, f + 1)).flatten(1));
    }

    // 2D: o.x2d_*.slice(1, l, l + 1) = matrix of link feature l between every pair of stocks -> [B, 1, N, N] -> [B, K2]
    for (int64_t l = 0; l < n_link_features_; ++l) {
      features.push_back(conv2d_price[l]->forward(o.x2d_price.slice(1, l, l + 1)).flatten(1));
      features.push_back(conv2d_volume[l]->forward(o.x2d_volume.slice(1, l, l + 1)).flatten(1));
    }

    // Portfolio percents go straight into the PPO input -> [B, N]
    features.push_back(o.x_portfolio);

    auto shared = cat(features, 1);         // [B, S]
    shared = mlp_shared->forward(shared);   // [B, 256]

    auto logits = actor->forward(shared);                  // [B, N]
    auto value  = critic->forward(shared).squeeze(-1);     // [B]
    return {logits, value};
  }

  // Convert allocation scores into nonnegative weights summing to one. [B, N] -> [B, N]
  Tensor action_to_weights(const Tensor& action) {
    return softmax(action, -1);
  }

  // [B, N] -> [B, N]
  Tensor configure_allocation_weights(const Tensor& weights) {
    if (Config::WINNER_TAKES_ALL) {
      const auto best_stock = weights.argmax(-1);
      return one_hot(best_stock, weights.size(-1)).to(weights.scalar_type());
    }
    return weights;
  }

  // Rollout-time: sample an action, no gradients.
  ActionOutput  act(const Observation& o, bool deterministic = false) {
    NoGradGuard ng;
    auto [mean, value] = forward(o);
    auto bounded_log_std = log_std.clamp(-5.0, 2.0);
    auto std = bounded_log_std.exp();
    auto action = deterministic ? mean : mean + std * torch::randn_like(mean);
    auto log_prob = -0.5 * ((action - mean) / std).pow(2) - bounded_log_std - 0.9189385332046727;
    Tensor weights_tensor = configure_allocation_weights(action_to_weights(action)).to(kCPU, kFloat32).contiguous();  // [1, N]
    vector<double> weights(weights_tensor.data_ptr<float>(), weights_tensor.data_ptr<float>() + weights_tensor.numel()); // [N]
    return {action, log_prob.sum(-1), value, weights};
  }

  // Update-time: re-score stored actions, gradients on.
  // Returns {log_prob [B], entropy [B], value [B]}.
  tuple<Tensor, Tensor, Tensor> evaluate_actions(const Observation& o, const Tensor& actions) {
    auto [mean, value] = forward(o);
    auto bounded_log_std = log_std.clamp(-5.0, 2.0);
    auto std = bounded_log_std.exp();
    auto log_prob = -0.5 * ((actions - mean) / std).pow(2) - bounded_log_std - 0.9189385332046727;
    auto entropy = torch::ones_like(value) * (bounded_log_std + 1.4189385332046727).sum();
    return {log_prob.sum(-1), entropy, value};
  }
};
TORCH_MODULE_IMPL(Agent, AgentImpl);
