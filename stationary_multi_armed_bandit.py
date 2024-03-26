import numpy as np

from agent_stationary import BanditAgent
from game_environment import MultiArmedGame


def train_bandit(game: MultiArmedGame, agent_bandit: BanditAgent, steps=1000):
    # Let's simulate 1000 steps of the bandit problem
    for _ in range(steps):
        action = agent_bandit.choose_action()
        reward = game.apply_action(action)
        print(reward)
        game.play_step()
        agent_bandit.update_estimates(action, reward)


if __name__ =='__main__':
    k = 10  # Number of actions (bandits)
    epsilon = 0.1  # Exploration probability
    steps = 1000
    file_path = './bandit_results.csv'

    agent_bandit = BanditAgent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False)

    train_bandit(game, agent_bandit, steps)
    data = agent_bandit.create_data(file_path, game.rewards.true_reward_probabilities)
    print(data.head(n=k))
