#pragma once

#include <memory>

#include "agent.hpp"

namespace quantsearch {

class PPOAgent : public TrainableAgent {
public:
	PPOAgent(std::string name, std::size_t stock_count,
	         std::vector<std::string> matrix_features, bool convolutional);
	~PPOAgent() override;

	const std::string& name() const override;
	Vector act(const AgentInput& input) override;
	void setTraining(bool training) override;
	void beginEpisode() override;
	void recordReward(double reward, bool terminal) override;
	TrainingStats updatePolicy() override;

private:
	struct Implementation;
	std::unique_ptr<Implementation> implementation_;
};

class PriceMatrixAgent final : public PPOAgent {
public:
	explicit PriceMatrixAgent(std::size_t stock_count);
};

class VolumeMatrixAgent final : public PPOAgent {
public:
	explicit VolumeMatrixAgent(std::size_t stock_count);
};

class DualMatrixAgent final : public PPOAgent {
public:
	explicit DualMatrixAgent(std::size_t stock_count);
};

class ConvolutionalAgent final : public PPOAgent {
public:
	explicit ConvolutionalAgent(std::size_t stock_count);
};

class ConvolutionalPriceAgent final : public PPOAgent {
public:
	explicit ConvolutionalPriceAgent(std::size_t stock_count);
};

class ConvolutionalVolumeAgent final : public PPOAgent {
public:
	explicit ConvolutionalVolumeAgent(std::size_t stock_count);
};

} // namespace quantsearch
