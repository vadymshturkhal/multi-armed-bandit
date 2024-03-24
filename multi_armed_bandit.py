import numpy as np

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

print("Estimated action values:", Q)
print("True action values:", true_reward_probabilities)
print("Number of times each action was chosen:", N)
