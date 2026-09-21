import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from NetworkTradingEnv import NetworkTradingEnv, SEED
from NetworkBuilder import NetworkBuilder
from NetworkFeatureExtractor import NetworkFeatureExtractor


class ArrObservationWrapper(gym.ObservationWrapper):
	"""
	Forces every observation that leaves the env to be `arr`,
	i.e. it always goes: raw_obs -> NetworkBuilder.calculate_dag -> dag
	                     -> NetworkFeatureExtractor.extract -> arr.

	No code path (check_env, Monitor, VecNormalize, PPO.predict, PPO.learn)
	can ever see the raw env observation, only `arr`.
	"""

	def __init__(self, env: gym.Env):
		super().__init__(env)

		# Run the conversion once (on a real reset obs) to discover arr's
		# actual shape/dtype, so observation_space is declared correctly.
		raw_obs, _ = env.reset(seed=SEED)
		arr = self._convert(raw_obs)

		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=arr.shape,
			dtype=arr.dtype,
		)

	@staticmethod
	def _convert(raw_obs):
		dag = NetworkBuilder.calculate_dag(raw_obs)
		_, arr = NetworkFeatureExtractor.extract(dag)
		return np.asarray(arr)

	def observation(self, obs):
		return self._convert(obs)


def make_env():
	env = NetworkTradingEnv()
	env = ArrObservationWrapper(env)
	env = Monitor(env)
	return env


def main():
	# check_env sees the wrapped env, so it validates against arr's
	# observation_space, not the raw env's.
	check_env(ArrObservationWrapper(NetworkTradingEnv()), warn=True)

	vec_env = DummyVecEnv([make_env])
	vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

	policy_kwargs = dict(
		features_extractor_class=NetworkFeatureExtractor,
		features_extractor_kwargs=dict(features_dim=128),
		net_arch=dict(pi=[64, 64], vf=[64, 64]),
	)

	model = PPO(
		policy="MlpPolicy",
		env=vec_env,
		policy_kwargs=policy_kwargs,
		n_steps=2048,
		batch_size=64,
		n_epochs=10,
		learning_rate=3e-4,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		ent_coef=0.0,
		verbose=1,
		seed=SEED,
	)

	model.learn(total_timesteps=200_000)

	model.save("ppo_trading")
	vec_env.save("vecnormalize_stats.pkl")


if __name__ == "__main__":
	main()