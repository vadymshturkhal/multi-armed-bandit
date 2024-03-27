from agent import NonStationaryAgent
from game_environment import MultiArmedGame
from plot_data import plot_average_rewards
from settings import nonstationary_bandit_data_average_reward
from utils import create_average_data


def train_bandit(game: MultiArmedGame, main_agent: NonStationaryAgent, support_agent: NonStationaryAgent, steps=1000):
    for _ in range(steps):
        action = main_agent.choose_action()
        support_action = support_agent.choose_action()

        reward = game.apply_action(action, bet=support_action)
        game.play_step()

        main_agent.update_estimates(action, reward)
        main_agent.update_points(reward)

        support_agent.update_estimates(support_action, reward)
        rewards.append(reward)
        last_bet = game.last_bet
        betting.append(last_bet)


if __name__ =='__main__':
    k = 4  # Number of actions (bandits)
    epsilon = 0.1  # Exploration probability
    alpha = 0.2
    steps = 100
    rewards = []
    betting = []

    main_agent = NonStationaryAgent(k, epsilon, alpha)
    game = MultiArmedGame(k, speed=60, is_rendering=False)
    support_agent = NonStationaryAgent(len(game.dealers[0].bet), epsilon, alpha)

    train_bandit(game, main_agent, support_agent, steps)
    create_average_data(nonstationary_bandit_data_average_reward, main_agent, rewards, betting)
    # plot_average_rewards(nonstationary_bandit_data_average_reward)
