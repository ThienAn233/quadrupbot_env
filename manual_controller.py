import pybullet as p
from stable_baselines3 import SAC
import numpy as np
from quad_multi_direct_v3_gym import Quadrup_env
import time as t


env = Quadrup_env(render_mode="human",terrainHeight=[0,0.05],terrain_type=2,ray_test=False)
model = SAC.load('SAC_gym_2024-02-20-15-13-15',device='cpu',print_system_info=True)
# p.setRealTimeSimulation(True)
# Id = p.addUserDebugParameter("sleep time", rangeMin = 0., rangeMax = 1/24) 

delay = 1./24.
x_corr = 10
y_corr = 0
v_change = 0.1
obs, info = env.reset()
env.target_dir_world[0] = np.array([x_corr,y_corr,1])
while True:
    # t.sleep(delay)
    env.stopper(delay)
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
    env.target_dir_world[0] = np.array([x_corr,y_corr,1])
    print(env.calculate_target(0))
    # delay = p.readUserDebugParameter(Id)
    if terminated:# or truncated:
        obs, info = env.reset()
        print("YOU FAILED")