import numpy as np
from agent import BanditAgent


k = 10  # Number of actions (bandits)
epsilon = 0.1  # Exploration probability
steps = 1000
file_path = './bandit_results.csv'

# True reward probabilities for each bandit
true_reward_probabilities = np.random.rand(k)

# Simulate pulling the bandit's lever
def bandit(a):
    return 1 if (np.random.rand() < true_reward_probabilities[a]) else 0

def train_bandit(agent_bandit, steps):
    # Let's simulate 1000 steps of the bandit problem
    for _ in range(steps):
        action = agent_bandit.choose_action()
        reward = bandit(action)
        agent_bandit.update_estimates(action, reward)


agent_bandit = BanditAgent(k, epsilon, true_reward_probabilities)
train_bandit(agent_bandit, steps)
data = agent_bandit.create_data(file_path=file_path)
print(data.head(n=k))

