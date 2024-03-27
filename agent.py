import numpy as np

from stationary.agent import Agent


class NonStationaryAgent(Agent):
    def __init__(self, k, epsilon, alpha):
        super().__init__(k, epsilon)
        self.alpha = alpha
        self.sigma = np.zeros(k)  # For the unbiased constant-step-size
    
    def choose_action(self):
        return super().choose_action()

    # Update the estimates of action values
    def update_estimates(self, action, reward):
        # Update the unbiased constant-step-size parameter
        self.sigma[action] += self.alpha * (1 - self.sigma[action])
        # Calculate the step size
        beta = self.alpha / self.sigma[action]
        # Update the estimate
        self.N[action] += 1
        self.Q[action] += beta * (reward - self.Q[action])
