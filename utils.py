import pandas as pd
import numpy as np


def create_bar_data(file_path: str, k: int, agent, true_reward_probabilities):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Action': np.arange(0, k),
        'Estimated Action Values': agent.Q,
        'True Action Values': true_reward_probabilities,
        'Number of Times Chosen': agent.N,
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results
