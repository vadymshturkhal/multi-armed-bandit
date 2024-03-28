from agent import NonStationaryAgent, NonStationaryAgentBet, NonStationaryAgentUCB
from game_environment import MultiArmedGame
from plot_data import plot_rewards
from settings import nonstationary_bandit_data_average_reward, BET
from utils import create_average_data


def train(game: MultiArmedGame, main_agent, support_agent, steps=1000):
    rewards.clear()
    betting.clear()
    for _ in range(steps):
        action = main_agent.choose_action()
        support_action = support_agent.choose_action()

        reward = game.apply_action(action, bet=support_action)
        game.play_step()
        last_bet = game.last_bet

        main_agent.update_estimates(action, reward - last_bet)

        support_agent.update_estimates(support_action, reward - last_bet)
        is_end = support_agent.update_points(last_bet, reward)

        rewards.append(reward)
        betting.append(last_bet)

        if is_end:
            # print(main_agent.points)
            break
    return main_agent.points

def start_epoch(main_agent, main_agent_params, support_agent, support_agent_params):
    cost = 0
    for _ in range(epochs):
        game = MultiArmedGame(k, speed=60, is_rendering=False)
        agent_instance = main_agent(*main_agent_params)
        support_agent_instance = support_agent(*support_agent_params)

        tax = train(game, agent_instance, support_agent_instance, steps)
        if tax > 0:
            cost += tax
        else:
            cost -= 1000
        create_average_data(nonstationary_bandit_data_average_reward, support_agent_instance, rewards, betting)

    # plot_rewards(nonstationary_bandit_data_average_reward)
    return cost


if __name__ =='__main__':
    k = 1  # Number of actions (bandits)
    epsilon = 0.2  # Exploration probability
    alpha = 0.5
    steps = 1000
    epochs = 1
    rewards = []
    betting = []

    main_agent = NonStationaryAgent
    main_agent_params = [k, epsilon, alpha]

    support_agent = NonStationaryAgentBet
    support_agent_params = [len(BET), epsilon, alpha]

    # main_agent = NonStationaryAgentUCB
    # main_agent_params = [k, alpha]

    # support_agent = NonStationaryAgentUCB
    # support_agent_params = [k, alpha]

    print(start_epoch(main_agent, main_agent_params, support_agent, support_agent_params))
    # plot_rewards(nonstationary_bandit_data_average_reward)
