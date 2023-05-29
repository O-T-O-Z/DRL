################################################################################
#
# Part B: Pitfall! game using synchronous advantage actor critic (A2C) approach
# Ömer Tarik Özyilmaz (s3951731) and Nikolai Herrmann (s3975207)
#
################################################################################

import gym
from gym.utils.play import play
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
from stable_baselines3.common.monitor import Monitor


STATUS = "eval"
GAME_NAME = "Pitfall-v0"
MODEL_NAME = GAME_NAME + "-a2c-20M"


def get_model():
    env = gym.make(GAME_NAME)
    env = make_vec_env(GAME_NAME, n_envs=4)
    env = VecFrameStack(env, n_stack=4)
    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"logdir/" + GAME_NAME,
        n_steps=75,
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    )
    return model, env


if __name__ == "__main__":
    if STATUS == "train":
        model, _ = get_model()
        rewards_history = model.learn(total_timesteps=20000000)
        model.save(MODEL_NAME)

    elif STATUS == "play":
        env = gym.make(GAME_NAME, render_mode="rgb_array")
        play(env, zoom=3)

    elif STATUS == "eval":
        model, env = get_model()
        model = model.load(MODEL_NAME)
        observation = env.reset()
        random = False

        ## Code below has been adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py#L11

        episode_rewards = []
        episode_lengths = []
        is_monitor_wrapped = (
            is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
        )
        n_envs = 4
        n_eval_episodes = 100

        episode_counts = np.zeros(n_envs, dtype="int")
        episode_count_targets = np.array(
            [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
        )

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        observations = env.reset()
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        while (episode_counts < episode_count_targets).any():
            if not random:
                actions, states = model.predict(
                    observations,
                    state=states,
                    episode_start=episode_starts,
                    deterministic=False,
                )
            else:
                actions = [env.action_space.sample() for x in range(4)]

            new_observations, rewards, dones, infos = env.step(actions)
            current_rewards += rewards
            current_lengths += 1
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if dones[i]:
                        print(current_rewards[i])
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            observations = new_observations

            env.render("human")

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(episode_rewards)
        print(mean_reward)
