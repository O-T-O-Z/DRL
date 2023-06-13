# Deep Reinforcement Learning *Catch* & Atari *Pitfall!*
This repository contains the code for our Deep Reinforcement Learning (DRL) project consisting of two tasks:
- **Task A**: Learning the game *Catch* with a Deep Q-Network (DQN).
- **Task B**: Learning the Atari 2600 *Pitfall!* game with a synchronous advantage actor critic (A2C) algorithm.

## Before you start
Please make sure to install all the required packages. You can do this by executing:
```
pip install -r requirements.txt
```
## Explanation of the contents
* `logdir/` contains the Tensorboard logging outputs.
* `partX_results/` contain results for tasks A and B.
* `catch.py` contains the code for the *Catch* game environment.
* `plotX.py` contains the code for plotting the data for tasks A and B.
* `partA.py` contains the code for training the *Catch* game-playing agent with the DQN algorithm.
* `partB.py` contains the code for training the *Pitfall!* game-playing agent with the A2C algorithm.
* `Pitfall-v0-a2c-10M.zip` & `` are the trained models on the *Pitfall!* game.