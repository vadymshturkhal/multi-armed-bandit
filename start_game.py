from agent import Agent
from game_environment import MultiArmedGame
from settings import stationary_bandit_detailed_run_data_filename
from utils import RunDataLogger


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
    epsilon = 0.2  # Exploration probability
    steps = 1000

    agent_bandit = Agent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False)
    run_data_logger = RunDataLogger(game.rewards.q_star)

    train_bandit(game, agent_bandit, run_data_logger, steps)
    run_data_logger.create_detailed_run_data(stationary_bandit_detailed_run_data_filename)
