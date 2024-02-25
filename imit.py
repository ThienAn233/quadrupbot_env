import numpy as np
import tempfile

import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

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
env_ref = DummyVecEnv([lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=0,ray_test=False,buffer_length=5,seed=rng.integers(0,100),reference=False))),
                       lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=1,ray_test=False,buffer_length=5,seed=rng.integers(0,100),reference=False))),
                       lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=2,ray_test=False,buffer_length=5,seed=rng.integers(0,100),reference=False))),
                       lambda: RolloutInfoWrapper(Monitor(gym.make("multi-v3",terrain_type=3,ray_test=False,buffer_length=5,seed=rng.integers(0,100),reference=False)))])

expert = load_policy(
    'sac',
    path='SAC_gym_2024-02-21-19-21-00',
    venv=env)

reward, _ = evaluate_policy(expert, env, 4)
print(f"Expert reward: {reward}")

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    batch_size=1024,
    rng=rng)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env_ref,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    dagger_trainer.train(2000)
reward, _ = evaluate_policy(dagger_trainer.policy, env, 20)
dagger_trainer.policy.save('/try_BC')
print(reward)
print('done')