from agent_non_stationary import NonStationaryBanditAgent
from game_environment import MultiArmedGame
from plot_data import plot_average_rewards
from settings import nonstationary_bandit_data_average_reward
from utils import create_average_data


def train_bandit(game: MultiArmedGame, agent_bandit: NonStationaryBanditAgent, steps=1000):
    for _ in range(steps):
        action = agent_bandit.choose_action()
        reward = game.apply_action(action)
        game.play_step()
        agent_bandit.update_estimates(action, reward)
        agent_bandit.update_points(reward)
        rewards_after_each_step.append(reward)


if __name__ =='__main__':
    k = 5  # Number of actions (bandits)
    epsilon = 0.1  # Exploration probability
    alpha = 0.2
    steps = 1000
    rewards_after_each_step = []

    agent_bandit = NonStationaryBanditAgent(k, epsilon, alpha)
    # agent_bandit = BanditAgent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False, is_change_probabilities=True)

    train_bandit(game, agent_bandit, steps)
    create_average_data(nonstationary_bandit_data_average_reward, rewards_after_each_step, agent_bandit)
    plot_average_rewards(nonstationary_bandit_data_average_reward)
