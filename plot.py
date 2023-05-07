
import numpy as np
import matplotlib.pyplot as plt

file_name = "group_06_catch_rewards_1683451096.932377.npy"

arr = np.load(file_name)
plt.plot(arr)
plt.xlabel("Testing Episode in 10s")
plt.ylabel("Average Reward")
plt.show()