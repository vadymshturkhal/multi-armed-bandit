# Multi-Armed Bandit Project

This repository contains a simulation of the Multi-Armed Bandit problem, implemented in Python. The project is designed to explore reinforcement learning concepts, specifically focusing on the ε-greedy strategy for a stationary agent. The game allows interaction and observation of the agent's learning process over time. Additionally, the project includes functionality for plotting data to visualize the agent's decisions and performance.

## Description

The Multi-Armed Bandit problem is a classic reinforcement learning scenario that models decision-making under uncertainty. In this project, I simulated a k-armed bandit machine, where each arm (action) has its own fixed but unknown probability of rewarding the player. The objective is to maximize the total reward over a series of arm pulls.

### Game

The game component allows users to interactively pull one of the k arms and observe the reward. The game is implemented using Pygame for a graphical interface, providing a visual and interactive experience.
The simulation features a dynamic rewards system to mimic changing environments, making the game more challenging and realistic. The `Rewards` class is responsible for managing the reward probabilities for each of the bandit's arms. Initially, each arm is assigned a random probability of delivering a reward. As the game progresses, these probabilities are updated to simulate a changing environment, ensuring that the optimal strategy evolves over time.

### Reinforcement Learning Stationary Agent

I implemented an ε-greedy reinforcement learning agent that learns to choose the arm with the highest expected reward over time, while still exploring other arms occasionally. The agent's learning process and decision-making strategy can be observed as it interacts with the bandit machine.

### Plotting Data

The project includes functionality to plot and visualize data, such as the number of times each arm has been chosen and the comparison between estimated and true probabilities of each arm. This visualization aids in understanding the learning and exploration-exploitation balance of the reinforcement learning agent.

## Installation

To run this project, you need Python 3 and the following Python libraries:
- Pygame
- NumPy
- Matplotlib
- Pandas

You can install the required libraries using pip:

```bash
pip install pygame numpy matplotlib pandas
```

## Running
To run the simulation with the reinforcement learning agent:
```bash
python start_game.py
```

## Viewing plots of the agent's performance and decisions:
```bash
python plot_data.py
```
