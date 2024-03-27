import pandas as pd
import matplotlib.pyplot as plt

from settings import stationary_bandit_data_bar_filename, stationary_bandit_data_average_reward


def plot_stationary_bar_data():
    # Load the results from a CSV file
    df = pd.read_csv(stationary_bandit_data_bar_filename)

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Action'], df['Number of Times Chosen'], color='skyblue')

    # Annotate bars with the estimated and true probabilities
    for bar, est_prob, true_prob in zip(bars, df['Estimated Action Values'], df['True Action Values']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'Est: {est_prob:.2f}\nTrue: {true_prob:.2f}', 
                    ha='center', va='bottom', color='black', fontsize=8)

    # Set the labels and title
    plt.xlabel('Arm Number')
    plt.ylabel('Times Chosen')
    plt.title('Number of Times Each Arm Has Been Chosen')
    plt.xticks(df['Action'])
    plt.show()

def plot_average_rewards(file_path: str):
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Plot the 'Average' column against the 'Step' column
    plt.figure(figsize=(10, 5))
    plt.plot(data['Steps'], data['Average'], label='Average Reward')
    
    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()

def plot_rewards(file_path: str):
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Plot the 'Average' column against the 'Step' column
    plt.figure(figsize=(10, 5))
    plt.plot(data['Steps'], data['Points'], label='Points')
    
    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('APoints')
    plt.title('Points Over Time')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    # plot_stationary_bar_data()
    plot_average_rewards(stationary_bandit_data_average_reward)
