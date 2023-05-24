import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from gym.spaces import Box
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

# Create the Atari environment
# env = make_atari_env('LunarLander-v2', n_envs=1, seed=0)
env = gym.make('ALE/Assault-v5')
env = Monitor(env, "logdir/")

# Create and train the SAC agent
model = A2C('CnnPolicy', env, verbose=1)
rewards_history = model.learn(total_timesteps=50000, progress_bar=True)
# # Evaluate the trained agent
# obs = env.reset()
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"Mean reward: {mean_reward:.2f}")

# # Plot the reward and loss
plot_results(["logdir/"], 50000, results_plotter.X_TIMESTEPS, "ALE/Assault-v5")
plt.show()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
    env.render()
    time.sleep(.01)
