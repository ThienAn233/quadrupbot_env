import sys
import numpy as np
import gym
import time as t

class Quadrup_env():
    def __init__(self,render_mode='human',*args, **kwargs):
        self.base_env = gym.make('BipedalWalker-v3',render_mode = render_mode)
        self.time_steps_in_current_episode = [0]
        self.action_space       =  self.base_env.action_space.shape[0]
        self.observation_space  = self.base_env.observation_space.shape[0]
        self.buffer_length = 1

        return
    def get_obs(self,*args, **kwargs):
        obs = self.base_env.reset()
        self.time_steps_in_current_episode = [0]
        return obs
    def sim(self,action,*args, **kwargs):
        obs, rew, done, trunc, _ = self.base_env.step(action)
        self.time_steps_in_current_episode[0] += 1
        if done|trunc:
            obs = self.base_env.reset()[0]
            self.time_steps_in_current_episode = [0]
        return obs,np.array([rew,0,0,0,0,0]),done | trunc
    def get_run_gait(self,*args, **kwargs):
        return
# env = Quadrup_env('human')
# obs = env.get_obs()[0][0]
# for i in range(1000):
#     action = np.random.normal(size=env.action_space)
#     obs,rew,done = env.sim(action)
#     print(env.time_steps_in_current_episode)
#     t.sleep(0.1)
    