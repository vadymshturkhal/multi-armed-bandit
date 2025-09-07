import pandas as pd
import matplotlib.pyplot as plt

from settings import stationary_bandit_brief_run_data_filename, stationary_bandit_detailed_run_data_filename


def plot_stationary_bar_data():
    # Load the results from a CSV file
    df = pd.read_csv(stationary_bandit_brief_run_data_filename)

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


def plot_violin_plot():
    # Load your CSV
    df = pd.read_csv(stationary_bandit_detailed_run_data_filename)

    # Group rewards by actionc
    rewards_by_action = [df[df["Action"] == a]["Reward"].values for a in sorted(df["Action"].unique())]

    # Plot violin plot
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(rewards_by_action, showmeans=False, showmedians=False, showextrema=False)

    # Color violins gray
    for pc in parts['bodies']:
        pc.set_facecolor("gray")
        pc.set_alpha(0.6)

    # Plot average reward per action as black line
    for i, action_rewards in enumerate(rewards_by_action, start=1):
        mean_reward = action_rewards.mean()
        plt.hlines(mean_reward, i-0.3, i+0.3, colors='black', linewidth=2)


    # Add reference line at global average
    plt.axhline(df["Reward"].mean(), color='red', linestyle='--', label="Global average reward")

    # Labels
    plt.xticks(range(1, len(rewards_by_action)+1), sorted(df["Action"].unique()))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.title("Violin plot of rewards per action")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_stationary_bar_data()
    plot_violin_plot()
