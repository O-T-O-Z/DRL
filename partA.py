import torch
from torch import nn
from torchsummary import torchsummary
from catch import CatchEnv
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque


# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


class DQN(nn.Module):
	def __init__(self, n_observations, n_actions):
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
			nn.Linear(512, n_actions),
		)

	def forward(self, x):
		return F.softmax(self.layers(x), dim=1)


class Trainer:

	def __init__(self, net_type, net_params, device, batch_size=32, epsilon=1.0, gamma=0.99, lr=0.01):
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = lr
		self.device = device

		self.target_net = net_type(*net_params).to(self.device)
		self.policy_net = net_type(*net_params).to(self.device)
		self.transfer_knowledge()

		self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=2e-4)
		self.criterion = nn.SmoothL1Loss()

		self.memory = deque([], 2000)

	def make_action(self, state):
		if self.epsilon > 0.01:
			self.epsilon -= 0.00009
		if random.random() > self.epsilon:
			state = state.reshape(1, 4, 84, 84)
			output = self.policy_net.forward(torch.Tensor(state).to(self.device))
			return output.argmax().item()
		else:
			return random.randint(0, 2)

	def save_trajectory(self, t):
		self.memory.append(t)

	def train(self):
		if len(self.memory) < self.batch_size:
			return  # no training possible

		train_data = random.sample(self.memory, self.batch_size)
		train_data_unzip = list(zip(*train_data))
		train_data_unzip = [np.array(x) for x in train_data_unzip]
		states, actions, next_states, rewards, terminals = train_data_unzip

		# Q(s, a)
		states = torch.Tensor(states).reshape(len(states), 4, 84, 84).to(self.device)
		model_output = self.policy_net(states)
		state_action_values = model_output[np.arange(self.batch_size), actions]

		# max(Q(s+1, a))
		#if not terminal:
		with torch.no_grad():
			next_states = torch.Tensor(next_states).reshape(len(next_states), 4, 84, 84).to(self.device)
			model_output = self.target_net(next_states)
			max_state_action_values = model_output.max(1)[0].detach().cpu().numpy()

		max_state_action_values[terminals] = 0
		td_target = torch.Tensor(rewards + (self.gamma * max_state_action_values)).to(self.device)
		loss = self.criterion(state_action_values.unsqueeze(1), td_target.unsqueeze(1))
		#clipping

		self.optimizer.zero_grad()
		loss.backward()
		# torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()

		return loss.item()

	def transfer_knowledge(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	episodes = 10000
	env = CatchEnv()
	terminate = False
	trainer = Trainer(DQN, (env.state_shape(), env.get_num_actions()), device)
	perform = []
	rewards_all = []
	rewards_avg = []

	for e in tqdm(range(episodes)):

		state = env.reset()
		losses = []

		while True:

			action = trainer.make_action(state)
			next_state, reward, terminate = env.step(action)

			trainer.save_trajectory((state, action, next_state, reward, terminate))

			state = next_state

			loss = trainer.train()
			# print(len(trainer.memory))

			if loss:
				losses.append(loss)
			if terminate:
				perform.append(np.mean(losses))
				rewards_all.append(reward)
				if len(rewards_all) > 100:
					mean = np.mean(rewards_all[-100:])
					print(mean)
					print(np.mean(losses[-100:]))
				break

		if e % 300 == 0:
			trainer.transfer_knowledge()

	print(perform)
	print(rewards_all)
	plt.plot(perform)
	plt.xlabel("episode")
	plt.ylabel("loss")
	plt.show()
	plt.plot(rewards_avg)
	plt.xlabel("episode")
	plt.ylabel("Average Reward")
	plt.show()


if __name__ == '__main__':
	main()
