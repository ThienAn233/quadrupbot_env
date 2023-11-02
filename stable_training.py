import quad_cnn_env_no_contact as qa
from stable_baselines3 import SAC
from gym_env_wrapper import CustomEnv
import os
# Instantiate the env   
log_path = "quadrupbot_env\\runs\\SAC\\" 
# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs\\SAC')
env = CustomEnv(qa)
# Define and Train the agent
model = SAC(policy="MlpPolicy",learning_rate=1e-4,env=env,verbose=1,tensorboard_log=log_path)
model.learn(350000,tb_log_name="SAC_tryout")
model.save('SAC_tryout')