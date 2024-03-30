from agent import Agent
from game_environment import MultiArmedGame
from plot_data import plot_stationary_bar_data
from settings import stationary_bandit_data_bar_filename
from utils import create_bar_data


def train_bandit(game: MultiArmedGame, agent_bandit: Agent, steps=1000):
    # Let's simulate 1000 steps of the bandit problem
    for _ in range(steps):
        action = agent_bandit.choose_action()
        reward = game.apply_action(action)
        game.play_step()
        agent_bandit.update_estimates(action, reward)
        rewards_after_each_step.append(reward)


if __name__ =='__main__':
    k = 10  # Number of actions (bandits)
    epsilon = 0.1  # Exploration probability
    steps = 1000
    rewards_after_each_step = []

    agent_bandit = Agent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False)

    train_bandit(game, agent_bandit, steps)
    create_bar_data(stationary_bandit_data_bar_filename, k, agent_bandit, game.rewards.true_reward_probabilities)
    plot_stationary_bar_data()
