import pandas as pd


class RunDataLogger:
    def __init__(self, true_rewards):
        self.true_rewards = true_rewards
        self.steps = []
        self.actions = []
        self.rewards = []
        self.Q_actions = []
        self.average_action_rewards = []
        self.average_general_rewards = []

    def add_step_data(self, step, action, reward, Q_action, average_action_reward, average_general_reward):
        self.steps.append(step)
        self.actions.append(action)
        self.rewards.append(reward)
        self.Q_actions.append(Q_action)
        self.average_action_rewards.append(average_action_reward)
        self.average_general_rewards.append(average_general_reward)

    def create_detailed_run_data(self, file_path):
        results = pd.DataFrame({
            'Step': self.steps,
            'Action': self.actions,
            'Reward': self.rewards,
            'Q[action]': self.Q_actions,
            'Average action reward': self.average_action_rewards,
            'Average general reward': self.average_general_rewards,
        })

        results = results.map(lambda x: f'{x:.4f}' if type(x) == float else x)

        results.to_csv(file_path, index=False)
        return results
