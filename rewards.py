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

class DealerRewards:
    def __init__(self, k:int=2):
        self.true_reward_probabilities = np.random.rand(k)
        self._case_a = [0.1, 0.9]
        self._case_b = [0.2, 0.8]
        self._cases = [self._case_a, self._case_b]
        self._cases_probabilities = [0.5, 0.5]

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        probability_case_index = np.random.choice(len(self._cases), size=1, p=self._cases_probabilities)[0]
        probability_case = self._cases[probability_case_index]

        return 1 if np.random.rand() < probability_case[action] else 0

    def change_reward_probabilities(self):
        self.true_reward_probabilities += np.random.normal(0, 0.01, size=self.true_reward_probabilities.shape)
        self.true_reward_probabilities = np.clip(self.true_reward_probabilities, 0, 1)
