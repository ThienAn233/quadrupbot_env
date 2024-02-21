import numpy as np
import argparse

import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout

gym.register(
    id="multi-v3",
    entry_point="quad_multi_direct_v3_gym:Quadrup_env",
    max_episode_steps=500)
env = DummyVecEnv([lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=0,ray_test=False)))for i in range(4)])

expert = load_policy(
    'sac',
    path='SAC_gym_2024-02-20-15-13-15',
    venv=env)

reward, _ = evaluate_policy(expert, env, 10)
print(f"Expert reward: {reward}")

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
print('done')