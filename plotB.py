import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

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
plt.title("Pitfall! training performance over 1 run")
plt.savefig(os.path.join(folder, "rewards_pitfall.pdf"), bbox_inches="tight")
plt.show()
plt.cla()


eval_file = "model_vs_random.txt"

data_eval = []
data_means = []

with open(os.path.join(folder, eval_file), "r") as f:
    for line in f.readlines():
        row = [float(x) for x in line.split(",")]
        data_eval.append(row)
        data_means.append(np.mean(row))


data_eval = pd.DataFrame({"A2C Model": data_eval[0], "Random": data_eval[1]})

my_colors = {'A2C Model': 'blue', 'Random': 'skyblue'}
ax = sns.boxplot(data=data_eval, palette=my_colors)



plt.ylabel("Average Reward")

plt.axhline(y=data_means[0], linestyle='--', color="green", label="A2C Model mean = " + str(data_means[0]), alpha=0.7)
plt.axhline(y=data_means[1], linestyle='--', color="dodgerblue", label="Random mean = " + str(data_means[1]), alpha=0.7)
plt.title('Pitfall! testing performance on 100 episodes')
plt.legend()
plt.savefig(os.path.join(folder, "eval_pitfall.pdf"), bbox_inches="tight")
plt.show()