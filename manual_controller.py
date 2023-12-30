import quad_multidirect_env as qa
import pybullet as p
from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_env_wrapper import CustomEnv
import time as t


env = CustomEnv(qa,render_mode="human")
model = SAC.load('SAC_tryout_strict_new',device='cpu',print_system_info=True)
# p.setRealTimeSimulation(True)


delay = 1./24.
x_corr = 10
y_corr = 0
v_change = 0.05
obs, info = env.reset()
env.env.target_dir_world[0] = np.array([x_corr,y_corr,1])
while True:
    t.sleep(delay)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    keys = p.getKeyboardEvents()
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW]&p.KEY_IS_DOWN:
        y_corr +=v_change
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW]&p.KEY_IS_DOWN:
        y_corr -=v_change
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW]&p.KEY_IS_DOWN:
        x_corr +=v_change
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW]&p.KEY_IS_DOWN:
        x_corr -=v_change
    env.env.target_dir_world[0] = np.array([x_corr,y_corr,1])
    print(env.env.calculate_target(0))
    if terminated:# or truncated:
        obs, info = env.reset()
        print("YOU FAILED")