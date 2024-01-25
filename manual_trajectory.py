import quad_multidirect_env as qa
import pybullet as p
from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_env_wrapper import CustomEnv
import time as t


env = CustomEnv(qa,render_mode="human")
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
model = SAC.load('SAC_tryout_strict_new',device='cpu',print_system_info=True)
# p.setRealTimeSimulation(True)

def stopper(time):
        start   = t.time()
        stop    = start
        while(stop<(start+time)):
            stop = t.time()
 
R       = 26**.5
pi      = np.pi
res     = 1024
phi     = np.linspace(0,2*pi,res)
x       = lambda phi: R*np.cos(phi)
y       = lambda phi: R*np.sin(phi)

delay = 1./24.
obs, info = env.reset()
t=0
env.env.target_dir_world[0] = np.array([x(phi[t]),y(phi[t]),0.297])
while True:
    # stopper(delay)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if np.linalg.norm(env.env.base_pos[0] - env.env.target_dir_world[0]) < 1:
        if t+1 >= res:
            t=0
        else:
            t +=1
        env.env.target_dir_world[0] = np.array([x(phi[t]),y(phi[t]),0.297])
    p.addUserDebugPoints([env.env.base_pos[0]],[[0,0,1]],lifeTime=10,pointSize=.5)
    p.addUserDebugPoints([(5/26**.5)*env.env.target_dir_world[0]],[[1,0,0]],lifeTime=10,pointSize=.5)
    keys = p.getKeyboardEvents()
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW]&p.KEY_IS_DOWN:
        obs, info = env.reset()
        print("YOU FAILED")