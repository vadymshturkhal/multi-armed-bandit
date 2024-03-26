import numpy as np


class Rewards:
    def __init__(self, k:int):
        self.true_reward_probabilities = np.random.rand(k)

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        return 1 if (np.random.rand() < self.true_reward_probabilities[action]) else 0