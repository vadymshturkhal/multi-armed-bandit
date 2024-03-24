import numpy as np
import pandas as pd


# Number of actions (bandits)
k = 10
# Exploration probability
epsilon = 0.1
# True reward probabilities for each bandit
true_reward_probabilities = np.random.rand(k)

# Initialize estimates of action values and action counts
Q = np.zeros(k)
N = np.zeros(k)

# Choose an action using epsilon-greedy strategy
def choose_action(Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q))  # Explore: choose a random action
    else:
        return np.argmax(Q)  # Exploit: choose the best current action

# Simulate pulling the bandit's lever
def bandit(a):
    return 1 if (np.random.rand() < true_reward_probabilities[a]) else 0

# Update the estimates of action values
def update_estimates(Q, N, action, reward):
    N[action] += 1
    Q[action] += (1 / N[action]) * (reward - Q[action])

# Let's simulate 1000 steps of the bandit problem
for _ in range(1000):
    action = choose_action(Q, epsilon)
    reward = bandit(action)
    update_estimates(Q, N, action, reward)

def clip_values(values: list) -> list:
    for i in range(len(values)):
        values[i] = f'{values[i]:.4f}'
    return values


# Creating a DataFrame to hold the results
results = pd.DataFrame({
    'Action': np.arange(1, k+1),
    'Estimated Action Values': clip_values(Q),
    'True Action Values': clip_values(true_reward_probabilities),
    'Number of Times Chosen': N
})

# Save the DataFrame to a CSV file
file_path = './bandit_results.csv'
results.to_csv(file_path, index=False)

results.head(), file_path
