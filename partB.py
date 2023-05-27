import gym
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from stable_baselines3.common.env_util import make_vec_env
from gym.spaces import Box
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

# Create the Atari environment
env = gym.make('Breakout-v0')
env = make_vec_env('Breakout-v0', n_envs=4)
env = VecFrameStack(env, n_stack=4)

# Create and train the SAC agent
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log="logdir/", n_steps=75, learning_rate=0.0007, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, rms_prop_eps=1e-5, use_rms_prop=True, use_sde=False, sde_sample_freq=-1, normalize_advantage=False, create_eval_env=False, policy_kwargs=None, device='auto', _init_setup_model=True)
rewards_history = model.learn(total_timesteps=10000000)
model.save("ppo_breakout")