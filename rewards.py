import numpy as np


class Rewards:
    def __init__(self, k:int):
        self._true_reward_probabilities = np.random.rand(k)

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        return 1 if (np.random.rand() < self._true_reward_probabilities[action]) else 0

    def change_reward_probabilities(self):
        self._true_reward_probabilities += np.random.normal(0, 0.01, size=self._true_reward_probabilities.shape)
        self._true_reward_probabilities = np.clip(self._true_reward_probabilities, 0, 1)

class DealerRewards:
    def __init__(self):
        self._case_a = [0.1, 0.3, 0.4, 0.2]
        self._case_b = [0.3, 0.4, 0.2, 0.1]
        self._cases = [self._case_a, self._case_b]
        self._cases_probabilities = [0.5, 0.5]
        self._current_probabilities = self._case_a

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        return 1 if np.random.rand() < self._current_probabilities[action - 1] else 0

    def change_reward_probabilities(self):
        probability_case_index = np.random.choice(len(self._cases), size=1)[0]
        self._current_probabilities = self._cases[probability_case_index]
