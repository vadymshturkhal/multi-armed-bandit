from agent import NonStationaryAgent, NonStationaryAgentUCB
from game_environment import MultiArmedGame
from plot_data import plot_rewards
from settings import nonstationary_bandit_data_average_reward
from utils import create_average_data


def train_bandit(game: MultiArmedGame, main_agent: NonStationaryAgent, support_agent: NonStationaryAgent, steps=1000):
    for _ in range(steps):
        action = main_agent.choose_action()
        support_action = support_agent.choose_action()

        reward = game.apply_action(action, bet=support_action)
        game.play_step()
        last_bet = game.last_bet

        main_agent.update_estimates(action, reward - last_bet)
        is_end = main_agent.update_points(last_bet, reward)

        support_agent.update_estimates(support_action, reward)
        rewards.append(reward)
        
        betting.append(last_bet)

        if is_end:
            # print(main_agent.points)
            break
    return main_agent.points


if __name__ =='__main__':
    k = 4  # Number of actions (bandits)
    epsilon = 0.1  # Exploration probability
    alpha = 0.8
    steps = 10000
    epochs = 40

    cost = 0
    for _ in range(epochs):
        rewards = []
        betting = []
        # main_agent = NonStationaryAgent(k, epsilon, alpha)
        main_agent = NonStationaryAgentUCB(k, alpha)
        game = MultiArmedGame(k, speed=60, is_rendering=False)
        support_agent = NonStationaryAgent(len(game.dealers[0].bet), epsilon, alpha)

        tax = train_bandit(game, main_agent, support_agent, steps)
        if tax > 0:
            cost += tax
        else:
            cost -= 1000
        create_average_data(nonstationary_bandit_data_average_reward, main_agent, rewards, betting)
        print(tax)
    print(cost)
