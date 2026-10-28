#pragma once

#include <torch/torch.h>
#include <string>
#include <tuple>
#include <vector>

#include "agent_convolutional.cpp"
#include "config.cpp"
#include "observation.cpp"

using namespace std;
using namespace torch;
using namespace torch::nn;

struct FinancialNetworkMatrixAgentImpl : Module {
	int64_t n_stocks_;
	Tensor log_std;
	Sequential mlp_shared{nullptr};
	Linear actor{nullptr};
	Linear critic{nullptr};

	FinancialNetworkMatrixAgentImpl(int64_t n_stocks, int64_t, int64_t): n_stocks_(n_stocks) {
		const int64_t input_features = 2 * n_stocks * n_stocks + n_stocks;
		mlp_shared = register_module("mlp_shared", Sequential(
			Linear(input_features, 512),
			ReLU(),
			Linear(512, 256),
			ReLU()
		));
		actor = register_module("actor", Linear(256, n_stocks));
		critic = register_module("critic", Linear(256, 1));
		log_std = register_parameter("log_std", torch::full({n_stocks}, -0.5));

		NoGradGuard ng;
		actor->weight.mul_(0.01);
		actor->bias.zero_();
	}

	tuple<Tensor, Tensor> forward(const Observation& o) {
		auto price_matrix = o.financial_network_price.flatten(1);
		auto volume_matrix = o.financial_network_volume.flatten(1);
		auto inputs = cat({price_matrix, volume_matrix, o.x_portfolio}, 1);
		auto shared = mlp_shared->forward(inputs);
		auto logits = actor->forward(shared);
		auto value = critic->forward(shared).squeeze(-1);
		return {logits, value};
	}

	Tensor action_to_weights(const Tensor& action) {
		return softmax(action, -1);
	}

	Tensor configure_allocation_weights(const Tensor& weights) {
		if (Config::WINNER_TAKES_ALL) {
			const auto best_stock = weights.argmax(-1);
			return one_hot(best_stock, weights.size(-1)).to(weights.scalar_type());
		}
		return weights;
	}

	ActionOutput act(const Observation& o, bool deterministic = false) {
		NoGradGuard ng;
		auto [mean, value] = forward(o);
		auto bounded_log_std = log_std.clamp(-5.0, 2.0);
		auto std = bounded_log_std.exp();
		auto action = deterministic ? mean : mean + std * torch::randn_like(mean);
		auto log_prob = -0.5 * ((action - mean) / std).pow(2) - bounded_log_std - 0.9189385332046727;
		Tensor weights_tensor = configure_allocation_weights(action_to_weights(action)).to(kCPU, kFloat32).contiguous();
		vector<double> weights(weights_tensor.data_ptr<float>(), weights_tensor.data_ptr<float>() + weights_tensor.numel());
		return {action, log_prob.sum(-1), value, weights};
	}

	tuple<Tensor, Tensor, Tensor> evaluate_actions(const Observation& o, const Tensor& actions) {
		auto [mean, value] = forward(o);
		auto bounded_log_std = log_std.clamp(-5.0, 2.0);
		auto std = bounded_log_std.exp();
		auto log_prob = -0.5 * ((actions - mean) / std).pow(2) - bounded_log_std - 0.9189385332046727;
		auto entropy = torch::ones_like(value) * (bounded_log_std + 1.4189385332046727).sum();
		return {log_prob.sum(-1), entropy, value};
	}
};

TORCH_MODULE_IMPL(FinancialNetworkMatrixAgent, FinancialNetworkMatrixAgentImpl);
