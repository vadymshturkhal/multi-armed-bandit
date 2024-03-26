import pandas as pd
import numpy as np


def create_bar_data(file_path: str, k: int, agent, true_reward_probabilities):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Action': np.arange(1, k+1),
        'Estimated Action Values': agent.Q,
        'True Action Values': true_reward_probabilities,
        'Number of Times Chosen': agent.N,
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results

def create_average_data(file_path: str, rewards, agent):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Steps': np.arange(1, len(rewards) + 1),
        'Rewards': rewards,
        'Average': np.cumsum(rewards) / np.arange(1, len(rewards) + 1),
        'Points': agent.rewards,
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results
