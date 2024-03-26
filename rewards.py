import numpy as np


class Rewards:
    def __init__(self, k:int):
        self._true_reward_probabilities = np.random.rand(k)

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        return 1 if (np.random.rand() < self.true_reward_probabilities[action]) else 0

    def change_reward_probabilities(self):
        self._true_reward_probabilities += np.random.normal(0, 0.01, size=self._true_reward_probabilities.shape)
        self._true_reward_probabilities = np.clip(self._true_reward_probabilities, 0, 1)
