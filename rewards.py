import numpy as np


class Rewards:
    def __init__(self, k:int):
        # True action values q*(a) drawn from N(0,1)
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=k)


    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        reward = np.random.normal(loc=self.q_star[action], scale=1.0, size=1)
        return reward[0]
