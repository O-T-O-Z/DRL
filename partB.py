import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import Box
from tqdm import tqdm
import numpy as np

# Create the Atari environment
env = make_atari_env('ALE/Skiing-v5', n_envs=1, seed=0)

# Create and train the SAC agent
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=5000)

# Evaluate the trained agent
obs = env.reset()
rewards = []
r2 = []
for i in tqdm(range(1000)):
    action, _ = model.predict(obs)
    action = int(action)  # Convert the action to an integer
    obs, reward, done, info = env.step([int(action)])
    rewards.append(reward)
    if done:
        r2.append(reward)
        obs = env.reset()

print(rewards, r2)
print(np.mean(rewards))
print(np.mean(r2))
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"Mean reward: {mean_reward:.2f}")