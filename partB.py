import gym
from gym.utils.play import play
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env


STATUS = "play"
GAME_NAME = "Pitfall-v0"
MODEL_NAME = GAME_NAME + "-a2c-10M"


def get_model():
    env = gym.make(GAME_NAME)
    env = make_vec_env(GAME_NAME, n_envs=4)
    env = VecFrameStack(env, n_stack=4)
    model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=f"logdir/" + GAME_NAME, n_steps=75, learning_rate=0.001, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01,
                max_grad_norm=0.5, rms_prop_eps=1e-5, use_rms_prop=True, use_sde=False, sde_sample_freq=-1, normalize_advantage=False, policy_kwargs=None, device='auto', _init_setup_model=True)
    return model, env

if __name__ == "__main__":
    if STATUS == "train":
        model, _ = get_model()
        rewards_history = model.learn(total_timesteps=10000000)
        model.save(MODEL_NAME)

    elif STATUS == "play":
        env = gym.make(GAME_NAME, render_mode='rgb_array')
        play(env, zoom=3)

    elif STATUS == "eval":
        model, env = get_model()
        model = model.load(MODEL_NAME)
        observation = env.reset()
        while True:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            env.render("human")
