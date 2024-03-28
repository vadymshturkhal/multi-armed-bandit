import pandas as pd
import numpy as np
import psycopg2
from db_setting import DB_NAME, USER, PASSWORD, HOST, PORT


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

def create_average_data(file_path: str, agent, rewards, betting):
    # Creating a DataFrame to hold the results
    results = pd.DataFrame({
        'Steps': np.arange(1, len(rewards) + 1),
        'Bet': betting,
        'Reward': rewards,
        'Points': agent.rewards,
        'Average': np.cumsum(rewards) / np.arange(1, len(rewards) + 1),
    })

    # Save the DataFrame to a CSV file
    results.to_csv(file_path, index=False)
    return results

def add_epoch_to_db(epoch, agent, rewards, betting):
    conn = psycopg2.connect(
        dbname=DB_NAME, 
        user=USER, 
        password=PASSWORD, 
        host=HOST, 
        port=PORT,
    )
    cur = conn.cursor()

    # Delete all previous epochs
    cur.execute("DELETE FROM epochs;")
    # Commit the transaction to make the changes permanent
    conn.commit()
    
    cur.execute("INSERT INTO epochs DEFAULT VALUES;")
    conn.commit()

    # Close cursor and connection
    cur.close()
    conn.close()
