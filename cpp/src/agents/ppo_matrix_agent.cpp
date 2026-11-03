#include "ppo_matrix_agent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <torch/torch.h>

#include "../config.hpp"

namespace quantsearch {

struct PPOConfig {
	static constexpr int EPOCHS = Config::PPO_EPOCHS;
	static constexpr int MINIBATCH_SIZE = Config::PPO_MINIBATCH_SIZE;
	static constexpr double gamma = Config::PPO_GAMMA;
	static constexpr double lambda = Config::PPO_LAMBDA;
	static constexpr double CLIP_RANGE = Config::PPO_CLIP_RANGE;
	static constexpr double VALUE_COEF = Config::PPO_VALUE_COEF;
	static constexpr double ENTROPY_COEF = Config::PPO_ENTROPY_COEF;
	static constexpr double MAX_GRAD_NORM = Config::PPO_MAX_GRAD_NORM;
};

namespace {

using torch::Tensor;

struct ActorCriticImpl : torch::nn::Module {
	ActorCriticImpl(std::int64_t stock_count, std::int64_t matrix_count,
	                std::int64_t input_rows, std::int64_t input_columns, bool convolutional)
		: stock_count_(stock_count), convolutional_(convolutional) {
		if (convolutional_) {
			conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(matrix_count, 16, 3).padding(1)));
			conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
			pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
			shared = register_module("shared", torch::nn::Sequential(
				torch::nn::Linear(32, 128),
				torch::nn::ReLU(),
				torch::nn::Linear(128, 128),
				torch::nn::ReLU()));
		} else {
			shared = register_module("shared", torch::nn::Sequential(
				torch::nn::Linear(matrix_count * input_rows * input_columns, 256),
				torch::nn::ReLU(),
				torch::nn::Linear(256, 128),
				torch::nn::ReLU()));
		}
		actor = register_module("actor", torch::nn::Linear(128, stock_count_));
		critic = register_module("critic", torch::nn::Linear(128, 1));
		log_std = register_parameter("log_std", torch::full({stock_count_}, -0.5));
		torch::NoGradGuard guard;
		actor->weight.mul_(0.01);
		actor->bias.zero_();
	}

	std::tuple<Tensor, Tensor> forward(const Tensor& input) {
		Tensor features;
		if (convolutional_) {
			features = pool->forward(torch::relu(conv2->forward(torch::relu(conv1->forward(input))))).flatten(1);
		} else {
			features = input.flatten(1);
		}
		features = shared->forward(features);
		return {actor->forward(features), critic->forward(features).squeeze(-1)};
	}

	std::int64_t stock_count_;
	bool convolutional_;
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::AdaptiveAvgPool2d pool{nullptr};
	torch::nn::Sequential shared{nullptr};
	torch::nn::Linear actor{nullptr}, critic{nullptr};
	Tensor log_std;
};

TORCH_MODULE(ActorCritic);

struct Transition {
	Tensor state;
	Tensor action;
	Tensor log_probability;
	Tensor value;
	double reward = 0.0;
	bool terminal = false;
	bool reward_recorded = false;
};

Tensor makeState(const AgentInput& input, const std::vector<std::string>& feature_names,
                 std::size_t stock_count, std::size_t input_rows, bool convolutional,
                 torch::Device device) {
	std::vector<float> values;
	values.reserve(feature_names.size() * input_rows * stock_count);
	for (const std::string& feature_name : feature_names) {
		const auto feature = input.find(feature_name);
		if (feature == input.end() || !std::holds_alternative<Matrix>(feature->second)) {
			throw std::invalid_argument("PPO agent requires matrix feature '" + feature_name + "'");
		}
		const Matrix& matrix = std::get<Matrix>(feature->second);
		if (matrix.size() != input_rows) throw std::invalid_argument("PPO agent received a matrix with the wrong row count");
		const std::size_t first_value = values.size();
		for (const auto& row : matrix) {
			if (row.size() != stock_count) throw std::invalid_argument("PPO agent received a matrix with the wrong column count");
			for (double value : row) values.push_back(static_cast<float>(value));
		}
		double mean = 0.0;
		for (std::size_t index = first_value; index < values.size(); ++index) mean += values[index];
		mean /= static_cast<double>(values.size() - first_value);
		double variance = 0.0;
		for (std::size_t index = first_value; index < values.size(); ++index) {
			const double difference = values[index] - mean;
			variance += difference * difference;
		}
		const double standard_deviation = std::sqrt(variance / static_cast<double>(values.size() - first_value));
		if (standard_deviation > 1e-8) {
			for (std::size_t index = first_value; index < values.size(); ++index) {
				values[index] = static_cast<float>((values[index] - mean) / standard_deviation);
			}
		} else {
			std::fill(values.begin() + static_cast<std::ptrdiff_t>(first_value), values.end(), 0.0f);
		}
	}
	Tensor state = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	return convolutional
		? state.reshape({1, static_cast<std::int64_t>(feature_names.size()),
		                 static_cast<std::int64_t>(input_rows), static_cast<std::int64_t>(stock_count)})
		: state.reshape({1, -1});
}

Tensor normalLogProbability(const Tensor& action, const Tensor& mean, const Tensor& log_std) {
	const Tensor bounded_log_std = log_std.clamp(-5.0, 2.0);
	const Tensor standard_deviation = bounded_log_std.exp();
	const Tensor element_log_probability =
		-0.5 * ((action - mean) / standard_deviation).pow(2) - bounded_log_std - 0.9189385332046727;
	return element_log_probability.sum(-1);
}

} // namespace

struct PPOAgent::Implementation {
	Implementation(std::string agent_name, std::size_t stocks, std::vector<std::string> features, bool use_convolutions)
		: name(std::move(agent_name)), stock_count(stocks), feature_names(std::move(features)),
		  input_rows(use_convolutions ? stocks : static_cast<std::size_t>(Config::OBS_LEN)),
		  convolutional(use_convolutions),
		  device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
		  model(static_cast<std::int64_t>(stocks),
			static_cast<std::int64_t>(feature_names.size()), static_cast<std::int64_t>(input_rows),
			static_cast<std::int64_t>(stocks), convolutional),
		  optimizer(model->parameters(), torch::optim::AdamOptions(Config::LEARNING_RATE).weight_decay(Config::WEIGHT_DECAY)) {
		if (stock_count == 0 || feature_names.empty()) throw std::invalid_argument("PPO agent requires stocks and matrix features");
		model->to(device);
		std::cout << "Initialized PPO agent '" << name << "' on device: "
		          << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << '\n';
	}

	std::string name;
	std::size_t stock_count;
	std::vector<std::string> feature_names;
	std::size_t input_rows;
	bool convolutional;
	torch::Device device;
	ActorCritic model;
	torch::optim::Adam optimizer;
	bool training = false;
	std::vector<Transition> transitions;
};

PPOAgent::PPOAgent(std::string name, std::size_t stock_count,
                   std::vector<std::string> matrix_features, bool convolutional)
	: implementation_(std::make_unique<Implementation>(
		  std::move(name), stock_count, std::move(matrix_features), convolutional)) {}

PPOAgent::~PPOAgent() = default;

const std::string& PPOAgent::name() const {
	return implementation_->name;
}

Vector PPOAgent::act(const AgentInput& input) {
	auto& implementation = *implementation_;
	Tensor state = makeState(input, implementation.feature_names, implementation.stock_count,
	                         implementation.input_rows, implementation.convolutional,
	                         implementation.device);
	Tensor action;
	Tensor log_probability;
	Tensor value;
	{
		torch::NoGradGuard guard;
		auto [mean, estimated_value] = implementation.model->forward(state);
		const Tensor bounded_log_std = implementation.model->log_std.clamp(-5.0, 2.0);
		action = implementation.training
			? mean + bounded_log_std.exp() * torch::randn_like(mean)
			: mean;
		log_probability = normalLogProbability(action, mean, implementation.model->log_std);
		value = estimated_value;
	}
	if (implementation.training) {
		implementation.transitions.push_back({state.detach(), action.detach(), log_probability.detach(),
		                                      value.detach(), 0.0, false});
	}
	Tensor weights = torch::softmax(action, -1);
	if (Config::WINNER_TAKES_ALL) {
		const Tensor best_stock = weights.argmax(-1);
		weights = torch::one_hot(best_stock, static_cast<std::int64_t>(implementation.stock_count))
		              .to(weights.scalar_type());
	}
	weights = weights.squeeze(0).to(torch::kCPU, torch::kFloat32).contiguous();
	const float* begin = weights.data_ptr<float>();
	return Vector(begin, begin + weights.numel());
}

void PPOAgent::setTraining(bool training) {
	implementation_->training = training;
}

void PPOAgent::beginEpisode() {
	implementation_->transitions.clear();
}

void PPOAgent::recordReward(double reward, bool terminal) {
	auto& transitions = implementation_->transitions;
	if (transitions.empty() || transitions.back().reward_recorded) {
		throw std::logic_error("PPO reward does not match an unscored agent action");
	}
	if (!std::isfinite(reward)) throw std::invalid_argument("PPO reward must be finite");
	transitions.back().reward = reward;
	transitions.back().terminal = terminal;
	transitions.back().reward_recorded = true;
}

TrainingStats PPOAgent::updatePolicy() {
	auto& implementation = *implementation_;
	auto& transitions = implementation.transitions;
	if (transitions.empty()) return {};
	for (const auto& transition : transitions) {
		if (!transition.reward_recorded) {
			throw std::logic_error("PPO episode contains an action without a recorded reward");
		}
	}

	const std::int64_t count = static_cast<std::int64_t>(transitions.size());
	std::vector<float> advantages(static_cast<std::size_t>(count));
	std::vector<float> returns(static_cast<std::size_t>(count));
	float gae = 0.0f;
	for (std::int64_t step = count - 1; step >= 0; --step) {
		const auto& transition = transitions[static_cast<std::size_t>(step)];
		const float next_value = step + 1 < count
			? transitions[static_cast<std::size_t>(step + 1)].value.item<float>() : 0.0f;
		const float next_non_terminal = transition.terminal ? 0.0f : 1.0f;
		const float value = transition.value.item<float>();
		const float delta = static_cast<float>(transition.reward) +
			static_cast<float>(PPOConfig::gamma) * next_value * next_non_terminal - value;
		gae = delta + static_cast<float>(PPOConfig::gamma * PPOConfig::lambda) * next_non_terminal * gae;
		advantages[static_cast<std::size_t>(step)] = gae;
		returns[static_cast<std::size_t>(step)] = gae + value;
	}

	std::vector<Tensor> states, actions, old_log_probabilities;
	states.reserve(transitions.size());
	actions.reserve(transitions.size());
	old_log_probabilities.reserve(transitions.size());
	for (const auto& transition : transitions) {
		states.push_back(transition.state);
		actions.push_back(transition.action);
		old_log_probabilities.push_back(transition.log_probability);
	}
	Tensor batch_states = torch::cat(states, 0);
	Tensor batch_actions = torch::cat(actions, 0);
	Tensor old_log_prob = torch::cat(old_log_probabilities, 0);
	Tensor batch_returns = torch::tensor(returns, torch::TensorOptions().dtype(torch::kFloat32).device(implementation.device));
	Tensor batch_advantages = torch::tensor(advantages, torch::TensorOptions().dtype(torch::kFloat32).device(implementation.device));
	if (count > 1) {
		batch_advantages = (batch_advantages - batch_advantages.mean()) /
		                   (batch_advantages.std() + 1e-8);
	} else {
		batch_advantages = batch_advantages - batch_advantages.mean();
	}

	TrainingStats stats;
	std::int64_t updates = 0;
	for (std::int64_t epoch = 0; epoch < PPOConfig::EPOCHS; ++epoch) {
		Tensor permutation = torch::randperm(count, torch::TensorOptions().dtype(torch::kInt64).device(implementation.device));
		for (std::int64_t start = 0; start < count; start += PPOConfig::MINIBATCH_SIZE) {
			const std::int64_t end = std::min(start + PPOConfig::MINIBATCH_SIZE, count);
			Tensor indices = permutation.slice(0, start, end);
			Tensor mini_states = batch_states.index_select(0, indices);
			Tensor mini_actions = batch_actions.index_select(0, indices);
			Tensor mini_old_log_prob = old_log_prob.index_select(0, indices);
			Tensor mini_returns = batch_returns.index_select(0, indices);
			Tensor mini_advantages = batch_advantages.index_select(0, indices);
			auto [mean, value] = implementation.model->forward(mini_states);
			Tensor log_probability = normalLogProbability(mini_actions, mean, implementation.model->log_std);
			Tensor ratio = (log_probability - mini_old_log_prob).exp();
			Tensor unclipped = ratio * mini_advantages;
			Tensor clipped = ratio.clamp(1.0 - PPOConfig::CLIP_RANGE, 1.0 + PPOConfig::CLIP_RANGE) * mini_advantages;
			Tensor policy_loss = -torch::minimum(unclipped, clipped).mean();
			Tensor value_loss = torch::mse_loss(value, mini_returns);
			Tensor entropy = (implementation.model->log_std.clamp(-5.0, 2.0) + 1.4189385332046727).sum();
			Tensor loss = policy_loss + PPOConfig::VALUE_COEF * value_loss - PPOConfig::ENTROPY_COEF * entropy;

			implementation.optimizer.zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(implementation.model->parameters(), PPOConfig::MAX_GRAD_NORM);
			implementation.optimizer.step();
			stats.policy_loss += policy_loss.item<double>();
			stats.value_loss += value_loss.item<double>();
			++updates;
		}
	}
	if (updates > 0) {
		stats.policy_loss /= static_cast<double>(updates);
		stats.value_loss /= static_cast<double>(updates);
	}
	transitions.clear();
	return stats;
}

PriceMatrixAgent::PriceMatrixAgent(std::size_t stock_count)
	: PPOAgent("price_matrix", stock_count, {"close_prices"}, false) {}

VolumeMatrixAgent::VolumeMatrixAgent(std::size_t stock_count)
	: PPOAgent("volume_matrix", stock_count, {"volumes"}, false) {}

DualMatrixAgent::DualMatrixAgent(std::size_t stock_count)
	: PPOAgent("dual_matrix", stock_count, {"close_prices", "volumes"}, false) {}

ConvolutionalAgent::ConvolutionalAgent(std::size_t stock_count)
	: PPOAgent("convolutional", stock_count, {"price_network", "volume_network"}, true) {}

ConvolutionalPriceAgent::ConvolutionalPriceAgent(std::size_t stock_count)
	: PPOAgent("convolutional_price", stock_count, {"price_network"}, true) {}

ConvolutionalVolumeAgent::ConvolutionalVolumeAgent(std::size_t stock_count)
	: PPOAgent("convolutional_volume", stock_count, {"volume_network"}, true) {}

} // namespace quantsearch
