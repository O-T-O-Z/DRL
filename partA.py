import torch
from torch import nn
from torchsummary import torchsummary

from catch import CatchEnv
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# def sample(self):
# 	return random.sample(self.memory, self.batch_size)


class DQN(nn.Module):
	def __init__(self, n_observations, n_actions, epsilon=0.1):
		super(DQN, self).__init__()
		self.epsilon = epsilon
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
		return F.softmax(self.layers(x), dim=1).argmax().item()

	def make_action(self, state):
		if random.random() > self.epsilon:
			# network selection
			state = state.reshape(1, 4, 84, 84)
			return self.forward(torch.Tensor(state))
		else:
			return random.randint(0, 2)


def main():
	episodes = 100
	batch_size = 32
	memory = []
	env = CatchEnv()
	terminate = False

	target_net = DQN(env.state_shape(), env.get_num_actions())
	policy_net = DQN(env.state_shape(), env.get_num_actions())
	target_net.load_state_dict(policy_net.state_dict())

	optimizer = optim.RMSprop(policy_net.parameters())
	criterion = nn.MSELoss()

	for e in range(episodes):
		# Initialize the environment and state
		state = env.reset()
		for t in range(100):
			# Select and perform an action
			action = policy_net.make_action(state)
			next_state, reward, terminate = env.step(action)

			# Store the transition in memory
			memory.append((state, action, next_state, reward))

			# Move to the next state
			state = next_state

			# Perform one step of the optimization (on the target network)
			#loss = optimize_model()
			if terminate:
				break
		# Update the target network, copying all weights and biases in DQN
		if e % 10 == 0:
			target_net.load_state_dict(policy_net.state_dict())


if __name__ == '__main__':
	main()
