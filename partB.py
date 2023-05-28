import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# games = ["Jamesbond-v0", "Riverraid-v0", "Pooyan-v0", "ChopperCommand-v0"]


env = gym.make("Pitfall-v0")
env = make_vec_env("Pitfall-v0", n_envs=4)
env = VecFrameStack(env, n_stack=4)

# Create and train the SAC agent
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=f"logdir/Pitfall-v0", n_steps=75, learning_rate=0.001, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, rms_prop_eps=1e-5, use_rms_prop=True, use_sde=False, sde_sample_freq=-1, normalize_advantage=False, create_eval_env=False, policy_kwargs=None, device='auto', _init_setup_model=True)
rewards_history = model.learn(total_timesteps=20000000)
model.save(f"Pitfall-v0-a2c-20M")

# for game in games:
#     # Create the Atari environment
#     env = gym.make(game)
#     env = make_vec_env(game, n_envs=4)
#     env = VecFrameStack(env, n_stack=4)
    
#     # Create and train the SAC agent
#     model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=f"logdir/{game}", n_steps=75, learning_rate=0.001, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, rms_prop_eps=1e-5, use_rms_prop=True, use_sde=False, sde_sample_freq=-1, normalize_advantage=False, create_eval_env=False, policy_kwargs=None, device='auto', _init_setup_model=True)
#     rewards_history = model.learn(total_timesteps=1000000)
#     model.save(f"{game}-a2c")
