
import numpy as np
import matplotlib.pyplot as plt

rewards = "partA_results/1683796437/group_06_catch_rewards_1683796437.npy"
losses = "losses_1683496002.npy"

arr = np.load(rewards, allow_pickle=True)
plt.plot(arr)
plt.xlabel("Testing Episode in 10s")
plt.ylabel("Average Reward")
plt.savefig(rewards.split(".")[0] + ".png")
plt.show()
