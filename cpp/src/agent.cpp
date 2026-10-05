#include <torch/torch.h>
#include <tuple>

using namespace std;
using namespace torch;
using namespace torch::nn;

struct Observation {
  Tensor x3d_price; // [B, n_stocks*n_link_features, D, H, W]
  Tensor x3d_volume;
  Tensor x2d_price; // [B, n_stocks*n_node_features, H, W]
  Tensor x2d_volume;
  Tensor x_portfolio; // [B, n_stocks]

  Observation to(Device d) {
    return {
      .x3d_price = x3d_price.to(d),
      .x3d_volume = x3d_volume.to(d),
      .x2d_price = x2d_price.to(d),
      .x2d_volume = x2d_volume.to(d),
      .x_portfolio = x_portfolio.to(d)
    };
  }

  Observation index(Tensor& idx) {
    return {
      .x3d_price = x3d_price.index_select(0, idx),
      .x3d_volume = x3d_volume.index_select(0, idx),
      .x2d_price = x2d_price.index_select(0, idx),
      .x2d_volume = x2d_volume.index_select(0, idx),
      .x_portfolio = x_portfolio.index_select(0, idx)
    };
  }

  static Observation stack(const vector<Observation> &v) {
    vector<Tensor> a, b, c, d, e;
    a.reserve(v.size()); b.reserve(v.size()); c.reserve(v.size());
    d.reserve(v.size()); e.reserve(v.size());
    for (const Observation& o : v) {
      a.push_back(o.x3d_price);  b.push_back(o.x3d_volume);
      c.push_back(o.x2d_price);  d.push_back(o.x2d_volume);
      e.push_back(o.x_portfolio);
    }
    return {
      .x3d_price = cat(a, 0),
      .x3d_volume = cat(b, 0),
      .x2d_price = cat(c, 0),
      .x2d_volume = cat(d, 0),
      .x_portfolio = cat(e, 0)
    };
  }
};

struct ActionOutput {
  Tensor action;    // [B, n_stocks] sampled allocation scores
  Tensor log_prob;  // [B]
  Tensor value;     // [B]
};

struct AgentImpl : Module {
  static constexpr int64_t kConvChannels = 64;
  static constexpr int64_t kPool3d       = 2;  // AdaptiveAvgPool3d({2,2,2})
  static constexpr int64_t kPool2d       = 3;  // AdaptiveAvgPool2d({3,3})
  static constexpr int64_t kFeat3d = kConvChannels * kPool3d * kPool3d * kPool3d; // 512
  static constexpr int64_t kFeat2d = kConvChannels * kPool2d * kPool2d;           // 576
  static constexpr int64_t kFeatVec = 64;

  int64_t n_stocks_;
  Tensor log_std;

  Sequential conv3d_price{nullptr};
  Sequential conv3d_volume{nullptr};
  Sequential conv2d_price{nullptr};
  Sequential conv2d_volume{nullptr};
  Sequential mlp_portfolio{nullptr};
  Sequential mlp_shared{nullptr};
  Linear actor{nullptr};
  Linear critic{nullptr};

  AgentImpl(int64_t n_stocks, int64_t n_node_features, int64_t n_link_features): n_stocks_(n_stocks) {

    int64_t c3 = n_stocks * n_link_features;  // channels of the 3D inputs
    int64_t c2 = n_stocks * n_node_features;  // channels of the 2D inputs

    auto make3d = [&] {
      return Sequential(
        Conv3d(Conv3dOptions(c3, 32, 3).padding(1)),
        ReLU(),
        Conv3d(Conv3dOptions(32, kConvChannels, 3).padding(1)),
        ReLU(),
        AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions({kPool3d, kPool3d, kPool3d}))
      );
    };
    auto make2d = [&] {
      return Sequential(
        Conv2d(Conv2dOptions(c2, 32, 3).padding(1)),
        ReLU(),
        Conv2d(Conv2dOptions(32, kConvChannels, 3).padding(1)),
        ReLU(),
        AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({kPool2d, kPool2d}))
      );
    };

    conv3d_price  = register_module("conv3d_price",  make3d());
    conv3d_volume = register_module("conv3d_volume", make3d());
    conv2d_price  = register_module("conv2d_price",  make2d());
    conv2d_volume = register_module("conv2d_volume", make2d());

    mlp_portfolio = register_module(
      "mlp_portfolio",
      Sequential(
          Linear(n_stocks, kFeatVec),
          ReLU(),
          Linear(kFeatVec, kFeatVec),
          ReLU()
        )
      );
    int64_t total_features = 2 * kFeat3d + 2 * kFeat2d + kFeatVec;  // 2240

    mlp_shared = register_module("mlp_shared", Sequential(Linear(total_features, 512), ReLU(), Linear(512, 256), ReLU()));

    actor  = register_module("actor",  Linear(256, n_stocks));
    critic = register_module("critic", Linear(256, 1));
    log_std = register_parameter("log_std", torch::full({n_stocks}, -0.5));

    NoGradGuard ng;
    actor->weight.mul_(0.01);
    actor->bias.zero_();
  }

  // Returns {allocation means [B, n_stocks], value [B]}.
  tuple<Tensor, Tensor> forward(const Observation& o) {
    auto f3a = conv3d_price->forward(o.x3d_price).flatten(1);
    auto f3b = conv3d_volume->forward(o.x3d_volume).flatten(1);
    auto f2a = conv2d_price->forward(o.x2d_price).flatten(1);
    auto f2b = conv2d_volume->forward(o.x2d_volume).flatten(1);
    auto fv  = mlp_portfolio->forward(o.x_portfolio);

    auto features = cat({f3a, f3b, f2a, f2b, fv}, 1);
    features = mlp_shared->forward(features);

    auto logits = actor->forward(features);
    auto value  = critic->forward(features).squeeze(-1);
    return {logits, value};
  }

  // Rollout-time: sample an action, no gradients.
  ActionOutput act(const Observation& o, bool deterministic = false) {
    NoGradGuard ng;
    auto [mean, value] = forward(o);
    auto bounded_log_std = log_std.clamp(-5.0, 2.0);
    auto std = bounded_log_std.exp();
    auto action = deterministic ? mean : mean + std * torch::randn_like(mean);
    auto log_prob = -0.5 * ((action - mean) / std).pow(2)
                    - bounded_log_std - 0.9189385332046727;
    return {action, log_prob.sum(-1), value};
  }

  // Update-time: re-score stored actions, gradients on.
  // Returns {log_prob [B], entropy [B], value [B]}.
  tuple<Tensor, Tensor, Tensor> evaluate_actions(const Observation& o, const Tensor& actions) {
    auto [mean, value] = forward(o);
    auto bounded_log_std = log_std.clamp(-5.0, 2.0);
    auto std = bounded_log_std.exp();
    auto log_prob = -0.5 * ((actions - mean) / std).pow(2)
                    - bounded_log_std - 0.9189385332046727;
    auto entropy = torch::ones_like(value) * (bounded_log_std + 1.4189385332046727).sum();
    return {log_prob.sum(-1), entropy, value};
  }

  // Convert allocation scores into nonnegative weights summing to one.
  Tensor action_to_weights(const Tensor& action) {
    return softmax(action, -1);
  }
};
TORCH_MODULE_IMPL(Agent, AgentImpl);
