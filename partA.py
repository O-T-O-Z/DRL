################################################################################
#
# Part A: catch game using value-based (DQN) approach 
# Ömer Tarik Özyilmaz (s3951731) and Nikolai Herrmann (s3975207)
#
################################################################################

import torch
import torch.optim as optim
from torch import nn
from torchsummary import summary

import random
import numpy as np
import time
import sys
import signal

from catch import CatchEnv

TRAINING_EPISODES = 2500

# For the general setup of the training the pytorch tutorial for the cart-pole game
# was examined. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):
    """
    Our Deep Q Network. Architecture adapted from:
    @article{mnih2015human,
        title={Human-level control through deep reinforcement learning},
        author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and  
                Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and  
                Graves, Alex and Riedmiller, Martin and Fidjeland, 
                Andreas K and Ostrovski, Georg and others},
        journal={nature},
        volume={518},
        number={7540},
        pages={529--533},
        year={2015},
        publisher={Nature Publishing Group}
    }
    """

    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        return self.layers(x) # don't use softmax here :)
    
    def __str__(self):
        return str(summary(self.layers, (4, 84, 84)))


class Buffer:
    """
    Replay memory buffer to store trajectories. The idea of using an array 
    instead of deque was taken from:
    https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DQN/replay_memory.py.
    The approach was further adapted by storing all trajectories in only two arrays.
    """

    def __init__(self, size, batch_size):
        self.size = size
        self.batch_size = batch_size
        self.buffer_idx = 0
        self.fill_size = 0
        self.states = np.zeros((size, 2, 4, 84, 84), dtype=np.float32)
        self.act_rew_term = np.zeros((size, 3), dtype=np.int64)

    def save_trajectory(self, state, action, next_state, reward, terminate):
        self.states[self.buffer_idx, 0, :, :, :] = state
        self.act_rew_term[self.buffer_idx, 0] = action
        self.states[self.buffer_idx, 1, :, :, :] = next_state
        self.act_rew_term[self.buffer_idx, 1] = reward
        self.act_rew_term[self.buffer_idx, 2] = terminate

        self.buffer_idx += 1
        if self.buffer_idx >= self.size:
            self.buffer_idx = 0
        if self.fill_size < self.size:
            self.fill_size += 1

    def sample_possible(self):
        return self.fill_size >= self.batch_size

    def sample_trajectories(self):
        sample_idxs = random.sample(range(self.fill_size), self.batch_size)

        return (self.states[sample_idxs, 0, :, :, :],
                self.act_rew_term[sample_idxs, 0],
                self.states[sample_idxs, 1, :, :, :],
                self.act_rew_term[sample_idxs, 1],
                self.act_rew_term[sample_idxs, 2])


class Trainer:
    """
    Our trainer: selects actions and keeps track of policy and target networks
    """

    def __init__(self, 
                 batch_size=32,
                 gamma=0.99, 
                 lr=1e-4, 
                 buffer_size=TRAINING_EPISODES * 10, # just save everything
                 criterion=nn.SmoothL1Loss(),
                 n_step_transfer=1032,
                 epsilon_start=1.0,
                 epsilon_min=0.1,
                 epsilon_subtract=1e-4,
                 ):
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.gamma = gamma
        self.lr = lr
        self.criterion = criterion
        self.n_step_transfer = n_step_transfer
        self.epsilon_min = epsilon_min
        self.epsilon_subtract = epsilon_subtract
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_net = DQN().to(self.device)
        self.policy_net = DQN().to(self.device)
        print(self.policy_net)
        self.transfer_knowledge()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        
        self.buffer = Buffer(buffer_size, batch_size)

    def use_policy(self, state):
        state_input = torch.Tensor(np.array([state])).to(self.device)
        output = self.policy_net.forward(state_input)
        return output.argmax().item()
    
    def make_action(self, state, use_own_policy=False):
        if use_own_policy:
            return self.use_policy(state)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_subtract
        if random.random() > self.epsilon:
            return self.use_policy(state)
        else:
            return random.randint(0, 2)

    def save_trajectory(self, t):
        self.buffer.save_trajectory(*t)

    def train(self):
        self.steps += 1

        if not self.buffer.sample_possible():
            return  # no training possible

        states, actions, next_states, rewards, terminals = self.buffer.sample_trajectories()

        # Q(s, a)
        states = torch.Tensor(states).to(self.device)
        model_output = self.policy_net(states)
        state_action_values = model_output[np.arange(self.batch_size), actions]

        # max(Q(s+1, a))
        next_states = torch.Tensor(next_states).to(self.device)
        model_output = self.target_net(next_states)
        max_state_action_values = model_output.max(dim=1)[0]

        terminals = torch.Tensor(terminals).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        max_state_action_values[np.argwhere(terminals.cpu() == 1)] = 0.0
        td_target = rewards + (self.gamma * max_state_action_values)

        loss = self.criterion(state_action_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.n_step_transfer == 0:
            self.transfer_knowledge()

        return loss.item()

    def transfer_knowledge(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


AVG_TESTING_REWARDS = []
TRAINING_LOSSES = []

def save_data(log=None):
    log = log if log else str(int(time.time()))
    np.save("group_06_catch_rewards_" + log, np.array(AVG_TESTING_REWARDS))
    np.save("losses_" + log, np.array(TRAINING_LOSSES))

def manual_early_stopping(sig, frame):
    print("Program stopped early")
    save_data()
    print("Validation data saved!")
    sys.exit(0)

signal.signal(signal.SIGINT, manual_early_stopping)


def main():
    assert TRAINING_EPISODES % 10 == 0
    episodes = TRAINING_EPISODES * 2

    env = CatchEnv()
    trainer = Trainer()
    validation = False
    testing_rewards = []
    mean_reward = None

    for e in range(episodes):

        state = env.reset()
        state = state.reshape(4, 84, 84)

        while True:

            action = trainer.make_action(state, use_own_policy=validation)
            next_state, reward, terminate = env.step(action)
            next_state = next_state.reshape(4, 84, 84)

            if not validation:
                trainer.save_trajectory((state, action, next_state, reward, terminate))
                loss = trainer.train()
                TRAINING_LOSSES.append(loss)

            state = next_state

            if terminate:
                if validation:
                    testing_rewards.append(reward)
                break

        if e % 10 == 0:
            if validation:
                mean_reward = np.mean(testing_rewards)
                AVG_TESTING_REWARDS.append(mean_reward)
                testing_rewards = []
            validation = not validation
            print("------------------------------------")

        print("Episode:", e, "/", episodes, ("Validation Reward: " + str(mean_reward) if not validation else ""))

    save_data()


if __name__ == '__main__':
    main()
