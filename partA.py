
import torch
import torch.optim as optim
from torch import nn
from torchsummary import summary

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from catch import CatchEnv


# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


# Network from https://daiwk.github.io/assets/dqn.pdf
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        print(summary(self.layers, (4, 84, 84)))

    def forward(self, x):
        return self.layers(x) # don't use softmax here :)


# Adapted from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DQN/replay_memory.py
class Buffer:

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

    def __init__(self, 
                 batch_size=32,
                 gamma=0.99, 
                 lr=1e-4, 
                 buffer_size=60000,
                 criterion=nn.SmoothL1Loss()
                 ):
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.gamma = gamma
        self.lr = lr
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_net = DQN().to(self.device)
        self.policy_net = DQN().to(self.device)
        self.transfer_knowledge()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        
        self.buffer = Buffer(buffer_size, batch_size)

    def make_action(self, state):
        if self.epsilon > 0.1:
            self.epsilon -= 1e-4
        if random.random() > self.epsilon:
            output = self.policy_net.forward(torch.Tensor(np.array([state])).to(self.device))
            return output.argmax().item()
        else:
            return random.randint(0, 2)

    def save_trajectory(self, t):
        self.buffer.save_trajectory(*t)

    def train(self):
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

        max_state_action_values[np.argwhere(torch.Tensor(terminals) == 1)] = 0.0
        rewards = torch.Tensor(rewards).to(self.device)
        td_target = rewards + (self.gamma * max_state_action_values)
        loss = self.criterion(state_action_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def transfer_knowledge(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def main():

    episodes = 30000
    env = CatchEnv()
    terminate = False
    trainer = Trainer()
    perform = []
    rewards_all = []
    steps = 0

    for e in tqdm(range(episodes)):

        state = env.reset()
        state = state.reshape(4, 84, 84)
        losses = []

        while True:
            steps += 1
            action = trainer.make_action(state)
            next_state, reward, terminate = env.step(action)
            next_state = next_state.reshape(4, 84, 84)

            trainer.save_trajectory((state, action, next_state, reward, terminate))

            state = next_state

            loss = trainer.train()

            if steps % 1000 == 0:
                trainer.transfer_knowledge()

            if loss:
                losses.append(loss)
            if terminate:
                perform.append(np.mean(losses))
                rewards_all.append(reward)
                mean = np.mean(rewards_all[-100:])
                print(mean)
                break

    print(perform)
    print(rewards_all)
    # plt.plot(perform)
    # plt.xlabel("episode")
    # plt.ylabel("loss")
    # plt.show()
    # plt.plot(rewards_avg)
    # plt.xlabel("episode")
    # plt.ylabel("Average Reward")
    # plt.show()


if __name__ == '__main__':
    main()
