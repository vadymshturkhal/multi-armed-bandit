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


def plot_agents_data(agent_first_filename, agent_second_filename):
    df1 = pd.read_csv(agent_first_filename)
    df2 = pd.read_csv(agent_second_filename)

    # Extract step and general reward
    steps1 = df1["Step"]
    general1 = df1["Average general reward"]

    steps2 = df2["Step"]
    general2 = df2["Average general reward"]

    # Plot comparison
    plt.figure(figsize=(8,5))
    plt.plot(steps1, general1, label=f"{agent_first_filename}")
    plt.plot(steps2, general2, label=f"{agent_second_filename}")
    plt.xlabel("Steps")
    plt.ylabel("Average general reward")
    plt.title("Comparison of Average General Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # plot_stationary_bar_data()
    plot_violin_plot()
