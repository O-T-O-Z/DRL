import numpy as np
import matplotlib.pyplot as plt
import os

folder = "partB_results"
file = os.path.join(folder, "20m_pitfall.csv")

data = []

with open(file, "r") as f:
    for line in f.readlines()[1:]:
        vals = line.strip().split(",")
        vals = [float(x) for x in vals]
        data.append(vals)

data = np.array(data)

plt.plot(data[:,1] / 1000000, data[:,2])
plt.xlabel("Timestep (in millions)")
plt.ylabel("Average Reward")
plt.title("Pitfall! performance over 1 run")
plt.savefig(os.path.join(folder, "rewards_pitfall.pdf"), bbox_inches="tight")
plt.show()