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

class NonStationaryAgentUCB(Agent):
    def __init__(self, k, alpha, c=2):
        """
        Initializes the agent with the specified number of actions (k), the step-size parameter (alpha),
        and the confidence level (c) for the UCB calculation.

        Parameters:
            k (int): The number of bandit arms.
            alpha (float): The step size for updating estimates.
            c (float): The confidence level for the UCB exploration term.
        """
        super().__init__(k)
        self.alpha = alpha
        self.c = c
        self.total_steps = 0

    def choose_action(self) -> int:
        """
        Selects an action using the Upper Confidence Bound (UCB) strategy.

        Returns:
            int: The index of the selected action.
        """
        self.total_steps += 1
        if np.min(self.N) == 0:  # To ensure each action is tried at least once
            return np.argmin(self.N)
        ucb_values = self.Q + self.c * np.sqrt((2 * np.log(self.total_steps)) / self.N)
        return np.argmax(ucb_values)
