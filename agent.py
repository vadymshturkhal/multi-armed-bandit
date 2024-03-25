import numpy as np


class BanditAgent:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        # Initialize estimates of action values and action counts
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    # Choose an action using epsilon-greedy strategy
    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))  # Explore: choose a random action
        else:
            return np.argmax(self.Q)  # Exploit: choose the best current action
