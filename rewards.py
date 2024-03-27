import numpy as np


class Rewards:
    def __init__(self, k:int):
        self.true_reward_probabilities = np.random.rand(k)

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        return 1 if (np.random.rand() < self.true_reward_probabilities[action]) else 0

    def change_reward_probabilities(self):
        self.true_reward_probabilities += np.random.normal(0, 0.01, size=self.true_reward_probabilities.shape)
        self.true_reward_probabilities = np.clip(self.true_reward_probabilities, 0, 1)

class DealerRewards:
    def __init__(self):
        self.rewards = [0, 1, 2, 4, 10]
        self._probabilities = [0.5, 0.25, 0.15, 0.08, 0.02]

    # Simulate pulling the bandit's lever
    def get_reward(self, bet):
        return bet * np.random.choice(self.rewards, p=self._probabilities)
