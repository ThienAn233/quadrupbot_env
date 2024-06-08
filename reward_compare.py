import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


SEED            = 36
num_eval        = 10
terrain_type    = [i for i in range(4)]
terrainHeight   = [i*0.01+0.01 for i in range (5)]
max_length      = 2000
model_name      = 'SAC_gym_2024-02-27-11-23-48'
tname        = ['Flat','Hills','Stairs','Combine']


data=pd.DataFrame(columns=['Reward','Episode length','Terrain type','Roughness'])


gym.register(
    id="multi-v3",
    entry_point="reward_compare_dummy_env:Quadrup_env",
    max_episode_steps=max_length,
    kwargs={'terrain_type':None})

def get_env(terrain_type,terrainHeight,max_length):
    rng     = np.random.default_rng(SEED)
    return DummyVecEnv([lambda: Monitor(gym.make("multi-v3",terrain_type=terrain_type,ray_test=False,terrainHeight=[0,terrainHeight],buffer_length=5,max_length=max_length,seed=rng.integers(0,100)))])
model = SAC.load(model_name,device='cpu')

for idx, name in enumerate(model.actor.modules()):
    print(idx, name)

for height in terrainHeight:
    for ttype in terrain_type:
        env = get_env(ttype,height,max_length)
        reward, eplen = evaluate_policy(model, env, num_eval,return_episode_rewards=True)
        new_data = pd.DataFrame({'Reward':reward,'Episode length':eplen,'Terrain type':[tname[ttype] for _ in range(num_eval)],'Roughness':[height for _ in range(num_eval)]})
        data = data.append(new_data)
        del env
        print(f"model {model_name} reward: {reward} and len: {eplen}")
        
        
clone = data.copy()
data.to_csv('reward_data_2.csv')