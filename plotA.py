import os
import numpy as np
import matplotlib.pyplot as plt

folder = os.path.join("partA_results", "4")
file = "group_06_catch_rewards_4.npy"

arr = np.load(os.path.join(folder, file), allow_pickle=True)
plt.plot(np.arange(0, len(arr)) * 10, arr)
plt.xlabel("Testing Episode")
plt.ylabel("Average Reward")
plt.title("Catch Performance")
plt.savefig(os.path.join(folder, "rewards_catch.pdf"), bbox_inches="tight")
plt.show()
