import pandas as pd
import numpy as np


def create_bar_data(file_path, k, Q, N, true_reward_probabilities):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Action': np.arange(1, k+1),
        'Estimated Action Values': clip_values(Q),
        'True Action Values': clip_values(true_reward_probabilities),
        'Number of Times Chosen': N
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results

def clip_values(values: list) -> list:
    for i in range(len(values)):
        values[i] = f'{values[i]:.4f}'
    return values