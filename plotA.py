import os
import numpy as np
import matplotlib.pyplot as plt

folder = "partA_results"

data = []

for i in range(1, 6):
    path = os.path.join(folder, str(i), f"group_06_catch_rewards_{i}.npy")
    data.append(np.load(path, allow_pickle=True))

data = np.array(data)

min_ = data.min(axis=0)
max_ = data.max(axis=0)
mean = data.mean(axis=0)
x = np.arange(0, len(mean)) * 10

plt.plot(x, mean)
plt.fill_between(x, min_, max_, alpha=0.3)
plt.xlabel("Testing Episode")
plt.ylabel("Average Reward")
plt.title("Catch testing performance over 5 runs")
plt.savefig(os.path.join(folder, "rewards_catch.pdf"), bbox_inches="tight")
plt.show()
