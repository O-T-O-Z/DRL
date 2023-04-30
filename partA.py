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


# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


class DQN(nn.Module):
	def __init__(self, n_observations, n_actions):
		super(DQN, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=3),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=3, stride=3),
			nn.Conv2d(16, 32, kernel_size=3),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=3, stride=3),
			nn.Conv2d(32, 64, kernel_size=3),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=3, stride=3),
			nn.Flatten(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions),
		)
		# print(torchsummary.summary(self.layers, (4,84,84)))

	def forward(self, x):
		return F.softmax(self.layers(x), dim=1)
		

class Trainer:

	def __init__(self, net_type, net_params, batch_size=32, epsilon=0.1, gamma=0.9, lr=0.01):
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = lr

		self.target_net = net_type(*net_params)
		self.policy_net = net_type(*net_params)
		self.transfer_knowledge()
		
		self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()

		self.memory = []

	def make_action(self, state):
		if random.random() > self.epsilon:
			state = state.reshape(1, 4, 84, 84)
			output = self.policy_net.forward(torch.Tensor(state))
			return output.argmax().item()
		else:
			return random.randint(0, 2)
		
	def save_trajectory(self, t):
		self.memory.append(t)

	def train(self):
		if len(self.memory) < self.batch_size:
			return # no training possible
		
		train_data = random.sample(self.memory, self.batch_size)
		train_data_unzip = list(zip(*train_data))

		states = np.array(train_data_unzip[0])
		actions = np.array(train_data_unzip[1])
		next_states = np.array(train_data_unzip[2])
		rewards = np.array(train_data_unzip[3])

		# Q(s, a)
		states = torch.Tensor(states).reshape(len(states), 4, 84, 84)
		model_output = self.policy_net.forward(states)
		state_action_values = model_output[np.arange(self.batch_size), actions]

		# max(Q(s+1, a))
		next_states = torch.Tensor(next_states).reshape(len(next_states), 4, 84, 84)
		model_output = self.target_net.forward(next_states)
		max_state_action_values = model_output.max(1)[0].detach().numpy()
		td_target = torch.Tensor(rewards + (self.gamma * max_state_action_values))
		
		loss = self.criterion(state_action_values, td_target)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()
		
	def transfer_knowledge(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())


def main():
	episodes = 300
	env = CatchEnv()
	terminate = False
	trainer = Trainer(DQN, (env.state_shape(), env.get_num_actions()))
	perform = []
	duration = []

	for e in tqdm(range(episodes)):

		state = env.reset()
		losses = []
		count = 0

		while True:
	
			action = trainer.make_action(state)
			next_state, reward, terminate = env.step(action)

			trainer.save_trajectory((state, action, next_state, reward))

			state = next_state

			loss = trainer.train()
			if loss:
				losses.append(loss)
			if terminate:
				perform.append(np.mean(losses))
				break

			count += 1
		duration.append(count)

		if e % 10 == 0:
			trainer.transfer_knowledge()

	print(perform)
	print(duration)
	plt.plot(perform)
	plt.xlabel("episode")
	plt.ylabel("loss")
	plt.show()
	plt.plot(duration)
	plt.xlabel("episode")
	plt.ylabel("Duration of game")
	plt.show()


if __name__ == '__main__':
	main()
