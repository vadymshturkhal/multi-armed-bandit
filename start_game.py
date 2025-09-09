from agent import Agent, GradientAgent, UCBAgent
from game_environment import MultiArmedGame
from settings import epsilon_agent_data_filename, gradient_agent_data_filename, ucb_agent_data_filename
from utils import RunDataLogger
from plot_data import plot_agents_data


def train_bandit(game: MultiArmedGame, agent_bandit: Agent, run_data_logger: RunDataLogger, steps=1000):
    for step in range(steps):
        action = agent_bandit.choose_action()
        reward = game.apply_action(action)
        game.play_step()
        agent_bandit.update_estimates(action, reward)

        # logger
        average_action_reward = agent_bandit.action_rewards[action] / agent_bandit.N[action]
        average_general_reward = sum(agent_bandit.action_rewards) / sum(agent_bandit.N)

        run_data_logger.add_step_data(step, action, reward, agent_bandit.Q[action], average_action_reward, average_general_reward)


if __name__ =='__main__':
    k = 10  # Number of actions (bandits)
    epsilon = 0.01  # Exploration probability
    alpha = 0.1
    steps = 10000

    print('training epsilon agent')
    agent_bandit = Agent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False)
    run_data_logger = RunDataLogger(game.rewards.q_star)
    train_bandit(game, agent_bandit, run_data_logger, steps)
    run_data_logger.create_detailed_run_data(epsilon_agent_data_filename)

    print('training gradient agent')
    agent_bandit = GradientAgent(k, alpha)
    run_data_logger = RunDataLogger(game.rewards.q_star)
    train_bandit(game, agent_bandit, run_data_logger, steps)
    run_data_logger.create_detailed_run_data(gradient_agent_data_filename)

    print('training ucb agent')
    agent_bandit = UCBAgent(k)
    run_data_logger = RunDataLogger(game.rewards.q_star)
    train_bandit(game, agent_bandit, run_data_logger, steps)
    run_data_logger.create_detailed_run_data(ucb_agent_data_filename)

    plot_agents_data(epsilon_agent_data_filename, gradient_agent_data_filename, ucb_agent_data_filename)
