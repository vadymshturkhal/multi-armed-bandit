import numpy as np
import pandas as pd


class BanditAgent:
    def __init__(self, k, epsilon, alpha, true_reward_probabilities):
        self.k = k
        self.epsilon = epsilon
        self.true_reward_probabilities = true_reward_probabilities
        self.alpha = alpha

        # Initialize estimates of action values and action counts
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.sigma = np.zeros(k)  # For the unbiased constant-step-size

    # Choose an action using epsilon-greedy strategy
    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))  # Explore: choose a random action
        else:
            return np.argmax(self.Q)  # Exploit: choose the best current action

    # Update the estimates of action values
    def update_estimates(self, action, reward):
        # Update the unbiased constant-step-size parameter
        self.sigma[action] += self.alpha * (1 - self.sigma[action])
        # Calculate the step size
        beta = self.alpha / self.sigma[action]
        # Update the estimate
        self.N[action] += 1
        self.Q[action] += beta * (reward - self.Q[action])

    def create_data(self, file_path):
        # Creating a DataFrame to hold the results
        results = pd.DataFrame({
            'Action': np.arange(1, self.k+1),
            'Estimated Action Values': self._clip_values(self.Q),
            'True Action Values': self._clip_values(self.true_reward_probabilities),
            'Number of Times Chosen': self.N
        })

        # Save the DataFrame to a CSV file
        results.to_csv(file_path, index=False)
        return results

    def _clip_values(self, values: list) -> list:
        for i in range(len(values)):
            values[i] = f'{values[i]:.4f}'
        return values