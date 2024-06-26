import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import time as t

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, base_env, *args, **kwargs):
        super().__init__()
        self.env = base_env.Quadrup_env(*args, **kwargs)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        print(self.env.action_space)
        print(self.env.observation_space)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.env.action_space,), dtype = np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low = -3.4e+38, high = 3.4e+38,
                                            shape = (self.env.buffer_length*self.env.observation_space,), dtype = np.float32)
    
    def stopper(self,time):
        start   = t.time()
        stop    = start
        while(stop<(start+time)):
            stop = t.time()
    
    def step(self, action,realtime=False, *args, **kwargs):
        action *= np.pi/4
        action = 0.2*action.reshape((1,-1))+0.8*self.env.get_run_gait(self.env.time_steps_in_current_episode[0])
        filtered_action = self.env.previous_pos*.8 + action*.2
        self.env.previous_pos = action
        self.env.time_steps_in_current_episode = [self.env.time_steps_in_current_episode[i]+1 for i in range(self.env.num_robot)]
        for _ in range(self.env.num_step):
            self.env.act(filtered_action)
            p.stepSimulation( physicsClientId=0)
            p.resetBasePositionAndOrientation(self.env.targetId,self.env.target_dir_world[0], [0,0,0,1], physicsClientId = 0)
        if realtime:
            self.stopper(1./24.)
        # if self.env.render_mode == "human":
            # self.env.viz()
        # Get obs
        self.env.update_buffer(0)
        reward = np.array(self.env.get_reward_value(0))
        ori, high, dis = np.sum(self.env.base_ori[0][-1])/np.linalg.norm(self.env.base_ori[0]), self.env.base_pos[0][-1], np.linalg.norm(self.env.target_dir_robot[0])
        terminated = (ori<.5) | (dis<0.5)
        truncated = self.env.time_steps_in_current_episode[0]>self.env.max_length
        info = {}
        return self.env.obs_buffer[0].flatten().astype('float32'), reward.sum(), bool(terminated), bool(truncated), info

    def reset(self, seed = 0, *args, **kwargs):
        self.env.sample_target(0)
        self.env.reset_buffer(0)
        self.env.time_steps_in_current_episode[0] = 0
        self.env.previous_pos[0] = np.zeros((len(self.env.jointId_list)))
        info = None
        return self.env.obs_buffer[0].flatten().astype('float32'), {}

    def render(self):
        self.env.viz()
        return

    def close(self):
        self.env.close(0)


# # # TEST CODE # # #
# import quad_multidirect_env as qa
# import time as t
# env = CustomEnv(qa,render_mode = 'human',max_length=500,terrainHeight   = [0. ,0.05])
# obs, info = env.reset()
# # print(obs.shape)
# for _ in range(5000):
#     action = np.random.random((env.env.action_space))
#     print(action)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if truncated or terminated:
#         obs, inf = env.reset()
#     t.sleep(0.05)
# env.close


# # # ENV CHECK # # #
# from stable_baselines3.common.env_checker import check_env
# import quad_multidirect_env as qa
# env = CustomEnv(qa,render_mode = 'human')
# # It will check your custom environment and output additional warnings if needed
# check_env(env)
# print('checked, no error!')


# # # TRAIN CHECK # # #
# import quad_multidirect_env as qa
# from stable_baselines3 import SAC
# # Instantiate the env
# env = CustomEnv(qa,render_mode = 'human')
# # # Define and Train the agent
# # model = SAC(policy="MlpPolicy",env=env,verbose=1,buffer_size=10)
# # model.learn(5000)
# # model.save('SAC_tryout')
# import time as t
# model = SAC.load('SAC_tryout_strict',device='cpu',print_system_info=True)
# obs, info = env.reset()
# while True:
#     t.sleep(0.05)
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(reward)
#     if terminated or truncated:
#         obs, info = env.reset()
