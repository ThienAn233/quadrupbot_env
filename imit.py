import numpy as np
import argparse

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout


parser = argparse.ArgumentParser(
                    prog='Imitation_learning',
                    description='Train a model',
                    epilog='Text at the bottom of help')

parser.add_argument('batch_size', metavar='N',default= int(2048) ,type=int, nargs=1,help='Batch size')
parser.add_argument('learning_rate', metavar='u',default= float(1e-4) ,type=float, nargs=1,help='Learning rate')
args = parser.parse_args()
# print(args.integers)
import os
print(os.getcwd())
gym.register(
    id="multi-v3",
    entry_point="quad_multi_direct_v3_gym:Quadrup_env",
    max_episode_steps=500)
env = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("multi-v3",terrain_type=i))for i in range(4)])

if __name__ == '__main__':
    # env = SubprocVecEnv([lambda: RolloutInfoWrapper(gym.make("multi-v3",terrain_type=i))for i in range(4)])

    expert = load_policy(
        'sac',
        path='SAC_v3_2024-02-11-17-56-36_500k_gSDE',
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
    