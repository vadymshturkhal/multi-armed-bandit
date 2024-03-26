from agent_stationary import BanditAgent
from game_environment import MultiArmedGame
from settings import stationary_bandit_data_bar_filename, stationary_bandit_data_average_reward
from utils import create_average_data, create_bar_data


def train_bandit(game: MultiArmedGame, agent_bandit: BanditAgent, steps=1000):
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

    agent_bandit = BanditAgent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False)

    train_bandit(game, agent_bandit, steps)
    create_bar_data(stationary_bandit_data_bar_filename, k, agent_bandit, game.rewards.true_reward_probabilities)
    create_average_data(stationary_bandit_data_average_reward, rewards_after_each_step)
