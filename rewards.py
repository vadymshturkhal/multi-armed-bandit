import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Rewards:
    def __init__(self, k:int):
        # True action values q*(a) drawn from N(0,1)
        self.k = k
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=k)
        self.samples = 0

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        reward = np.random.normal(loc=self.q_star[action], scale=1.0, size=1)
        self.samples += 1
        return reward[0]

    def plot_true_values(self):
        # For plotting: generate samples from each arm's distribution
        rewards = [np.random.normal(loc=q, scale=1.0, size=self.samples) for q in self.q_star]

        # Plot violin plot
        plt.figure(figsize=(10, 6))
        parts = plt.violinplot(rewards, showmeans=False, showmedians=False, showextrema=False)

        # Color violins
        for pc in parts['bodies']:
            pc.set_facecolor("gray")
            pc.set_alpha(0.6)

        # Plot the true means q*(a) as black horizontal lines
        for i, q in enumerate(self.q_star, start=1):
            plt.hlines(q, i-0.3, i+0.3, colors='black', linewidth=2)

        # Add reference line at 0
        plt.axhline(0, color='black', linestyle='--')

        # Labels
        plt.xticks(range(1, self.k+1))
        plt.xlabel("Action")
        plt.ylabel("Reward distribution")
        plt.title(f"Example {self.k}-armed bandit problem")
        plt.show()

    def plot_combined(self, csv_path):
        # --- True bandit values ---
        rewards_true = [np.random.normal(loc=q, scale=1.0, size=self.samples) for q in self.q_star]

        # --- Empirical data from CSV ---
        df = pd.read_csv(csv_path)
        df["Reward"] = pd.to_numeric(df["Reward"], errors="coerce")
        df = df.dropna(subset=["Reward"])

        actions = sorted(df["Action"].unique())
        rewards_by_action = [df[df["Action"] == a]["Reward"].values for a in actions]

        # --- Make 2 subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        # ----- Left: True distributions -----
        parts = axes[0].violinplot(rewards_true, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor("gray")
            pc.set_alpha(0.6)

        # True means
        for i, q in enumerate(self.q_star, start=1):
            axes[0].hlines(q, i-0.3, i+0.3, colors='black', linewidth=2)

        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].set_xticks(range(1, self.k+1))
        axes[0].set_xlabel("Action")
        axes[0].set_ylabel("Reward distribution")
        axes[0].set_title(f"True {self.k}-armed bandit problem")

        # ----- Right: Empirical violin plot -----
        parts = axes[1].violinplot(rewards_by_action, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor("gray")
            pc.set_alpha(0.6)

        # Mean rewards
        for i, action_rewards in enumerate(rewards_by_action, start=1):
            mean_reward = action_rewards.mean()
            axes[1].hlines(mean_reward, i-0.3, i+0.3, colors='black', linewidth=2)

        axes[1].axhline(0, color='black', linestyle='--')

        # Global average line
        axes[1].axhline(df["Reward"].mean(), color='red', linestyle='--', label="Global average reward")

        axes[1].set_xticks(range(1, len(rewards_by_action)+1))
        axes[1].set_xticklabels(actions, rotation=30)
        axes[1].set_xlabel("Action")
        axes[1].set_title("Empirical rewards per action")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
