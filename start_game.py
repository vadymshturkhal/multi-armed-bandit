from agent import Agent
from game_environment import MultiArmedGame
from settings import stationary_bandit_data_bar_filename, stationary_bandit_data_average_reward
from utils import create_bar_data


def train_bandit(game: MultiArmedGame, agent_bandit: Agent, steps=1000):
    with open(stationary_bandit_data_average_reward, "w") as f:
        f.write('Step, Action, Reward, Q[action], Average action reward, Average general reward\n')
        # Let's simulate 1000 steps of the bandit problem
        for step in range(steps):
            action = agent_bandit.choose_action()
            reward = game.apply_action(action)
            game.play_step()
            agent_bandit.update_estimates(action, reward)

            # logger
            average_action_reward = agent_bandit.action_rewards[action] / agent_bandit.N[action]
            average_general_reward = sum(agent_bandit.action_rewards) / sum(agent_bandit.N)
            f.write(f'{step}, {action}, {reward}, {agent_bandit.Q[action]:.4f}, {average_action_reward:.4f}, {average_general_reward:.4f}\n')
    

if __name__ =='__main__':
    k = 10  # Number of actions (bandits)
    epsilon = 0.2  # Exploration probability
    steps = 1000

    agent_bandit = Agent(k, epsilon)
    game = MultiArmedGame(k, speed=60, is_rendering=False)

    train_bandit(game, agent_bandit, steps)
    create_bar_data(stationary_bandit_data_bar_filename, k, agent_bandit, game.rewards.true_reward_probabilities)
