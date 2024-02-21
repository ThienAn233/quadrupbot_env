import numpy as np
import argparse

import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.gail import GAIL

SEED = 42
gym.register(
    id="multi-v3",
    entry_point="quad_multi_direct_v3_gym:Quadrup_env",
    max_episode_steps=500,
    kwargs={'terrain_type':None})
rng     = np.random.default_rng(SEED)
env     = DummyVecEnv([lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=0,ray_test=False,buffer_length=5,seed=rng.integers(0,100)))),
                       lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=1,ray_test=False,buffer_length=5,seed=rng.integers(0,100)))),
                       lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=2,ray_test=False,buffer_length=5,seed=rng.integers(0,100)))),
                       lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=3,ray_test=False,buffer_length=5,seed=rng.integers(0,100))))])

expert = load_policy(
    'sac',
    path='SAC_gym_2024-02-20-15-13-15',
    venv=env)

reward, _ = evaluate_policy(expert, env, 4)
print(f"Expert reward: {reward}")

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    rng=rng)

learner = SAC(
    policy="MlpPolicy",
    buffer_size=10,
    batch_size=2048,
    learning_rate= 3e-4,
    env=env,
    verbose=1,
    seed=SEED)

reward_before, _ = evaluate_policy(learner, env, 4)

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net)

# gail_trainer.train(100)

reward_after, _ = evaluate_policy(learner, env, 4)
print(f"Learner reward before training: {reward_before}")
print(f"Learner reward after training: {reward_after}")

print('done')