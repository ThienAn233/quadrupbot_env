import quad_multidirect_env as qa
from stable_baselines3 import SAC
from quad_multi_direct_v3_gym import Quadrup_env
import time as t
import numpy as np
from matplotlib import pyplot as plt
env = Quadrup_env(max_length=300,render_mode=None,ray_test=False,terrain_type=0,terrainHeight=[0, 0.0])
model = SAC.load('SAC_gym_2024-02-20-15-13-15',device='cpu',print_system_info=True)
obs, info = env.reset()
num_sample = 10
batch_cosine = []
batch_angle = []
batch_distance = []
for i in range(num_sample):
    data_cosine = []
    data_angle = []
    data_distance =[]
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        data_cosine += [env.target_dir_robot[0][0]/np.linalg.norm(env.target_dir_robot[0])]
        data_angle += [np.arctan2(env.target_dir_robot[0][1],env.target_dir_robot[0][0])]
        data_distance += [np.linalg.norm(env.target_dir_robot[0])]
        if terminated or truncated:
            obs, info = env.reset()
            break
    batch_cosine += [data_cosine]
    batch_angle += [data_angle]
    batch_distance += [data_distance]
batch_cosine = np.array(batch_cosine).T
batch_angle = np.array(batch_angle).T
batch_distance = np.array(batch_distance).T
x_values = np.array([0, 100, 200, 300, 400, 500])
x_ticks = np.round(x_values*(1/24),2)
plt.xticks(x_values,x_ticks)
plt.title("Khoảng cách giữa robot và mục tiêu")
plt.ylabel("d")
plt.xlabel("Thời gian (s)")
# plt.plot(batch_cosine)
# plt.plot(batch_angle)
plt.plot(batch_distance)
plt.show()