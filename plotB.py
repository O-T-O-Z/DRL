import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

folder = "partB_results"
file = os.path.join(folder, "10m_pitfall.csv")
file2 = os.path.join(folder, "10m_pitfall_adam.csv")

def read_csv(file_name):
    data = []

    with open(file_name, "r") as f:
        for line in f.readlines()[1:]:
            vals = line.strip().split(",")
            vals = [float(x) for x in vals]
            data.append(vals)

    return np.array(data)

data_rmsprop = read_csv(file)
data_adam = read_csv(file2)

plt.plot(data_rmsprop[:, 1] / 1000000, data_rmsprop[:, 2], label="RMSprop")
plt.plot(data_adam[:, 1] / 1000000, data_adam[:, 2], label="Adam")
plt.xlabel("Timestep (in millions)")
plt.ylabel("Average Reward")
plt.title("Pitfall! training performance over 1 run")
plt.legend()
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


data_eval = pd.DataFrame({"A2C Model (Adam)": data_eval[0], "A2C Model (RMSprop)": data_eval[1], "Random": data_eval[2]})

my_colors = {"A2C Model (Adam)": "orange", "A2C Model (RMSprop)": "blue", "Random": "skyblue"}
ax = sns.boxplot(data=data_eval, palette=my_colors)


plt.ylabel("Average Reward")

plt.axhline(
    y=data_means[0],
    linestyle="--",
    color="orange",
    label="A2C Model (Adam) mean = " + str(data_means[0]),
    alpha=0.7,
)
plt.axhline(
    y=data_means[1],
    linestyle="--",
    color="blue",
    label="A2C Model (RMSprop) mean = " + str(data_means[1]),
    alpha=0.7,
)
plt.axhline(
    y=data_means[2],
    linestyle="--",
    color="skyblue",
    label="Random mean = " + str(data_means[2]),
    alpha=0.7,
)
plt.title("Pitfall! testing performance on 100 episodes")
plt.legend()
plt.savefig(os.path.join(folder, "eval_pitfall.pdf"), bbox_inches="tight")
plt.show()
