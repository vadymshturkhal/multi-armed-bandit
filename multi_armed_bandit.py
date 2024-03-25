import numpy as np
from agent import BanditAgent
from game_environment import MultiArmedGame
import matplotlib.pyplot as plt


# Simulate pulling the bandit's lever
def bandit(a):
    return 1 if (np.random.rand() < true_reward_probabilities[a]) else 0

def train_bandit(game, agent_bandit, steps):
    # Let's simulate 1000 steps of the bandit problem
    optimal_action_count = np.zeros(steps)
    for step in range(steps):
        action = agent_bandit.choose_action()
        game.apply_action(action)
        game.play_step()
        reward = bandit(action)
        agent_bandit.update_estimates(action, reward)

        if action == optimal_action:
            optimal_action_count[step] = 1

    return np.cumsum(optimal_action_count) / np.arange(1, steps + 1)

# Define a function to plot the results
def plot_results(percentages, epsilon):
    plt.figure(figsize=(10, 5))
    # plt.plot(range(steps), percentages, label='ε-greedy')
    plt.plot(percentages, label=f'Realistic, ε-greedy (ε={epsilon})')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Percentage of Optimal Actions Taken Over Time')
    plt.legend()
    plt.show()

if __name__ =='__main__':
    k = 10  # Number of actions (bandits)
    epsilon = 0.1  # Exploration probability
    steps = 1000
    file_path = './bandit_results.csv'

    # True reward probabilities for each bandit
    true_reward_probabilities = np.random.rand(k)
    optimal_action = np.argmax(true_reward_probabilities)

    agent_bandit = BanditAgent(k, epsilon, true_reward_probabilities)
    game = MultiArmedGame(k, true_reward_probabilities, speed=60, is_rendering=False)

    percentage_optimal_action = train_bandit(game, agent_bandit, steps=1000)
    data = agent_bandit.create_data(file_path=file_path)
    print(data.head(n=k))

    # Plot learning curve
    plot_results(percentage_optimal_action, epsilon)


