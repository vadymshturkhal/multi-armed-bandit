import numpy as np


class Agent:
    """
    A class representing an agent that interacts with a stationary k-armed bandit environment.

    The agent uses an epsilon-greedy strategy for action selection. This means that with
    probability epsilon, the agent will explore and select an action at random. With
    probability 1 - epsilon, the agent will exploit its current knowledge and select the
    action with the highest estimated value.

    Attributes:
        k (int): The number of actions, i.e., the arms of the bandit.
        epsilon (float): The probability of selecting a random action; in [0, 1].
        Q (ndarray): The estimated value of each action initialized to zeros.
        N (ndarray): The count of the number of times each action has been selected.

    Methods:
        choose_action(): Selects an action using the epsilon-greedy strategy.
        update_estimates(action, reward): Updates the estimated values (Q) based on the received reward.
    """

    def __init__(self, k, epsilon=0.1):
        """
        Initializes the BanditAgent with the specified number of actions (k) and the exploration rate (epsilon).

        Parameters:
            k (int): The number of bandit arms.
            epsilon (float): The probability of exploration (choosing a random action).
        """
        self.k = k
        self.epsilon = epsilon

        # Initialize estimates of action values and action counts
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.action_rewards = np.zeros(k)


    def choose_action(self) -> int:
        """
        Selects an action using the epsilon-greedy strategy.

        With probability epsilon, a random action is chosen. Otherwise, the action with the highest
        estimated value is selected.

        Returns:
            int: The index of the selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))  # Explore: choose a random action
        else:
            return np.argmax(self.Q)  # Exploit: choose the best current action


    def update_estimates(self, action, reward):
        """
        Updates the action value estimates (Q) for a specific action based on the received reward.

        Parameters:
            action (int): The index of the action taken.
            reward (float): The reward received from taking the action.
        """
        self.N[action] += 1
        self.action_rewards[action] += reward

        #  Used Sample-Average as the action-value estimate
        # self.Q[action] = self.action_rewards[action] / self.N[action]

        #  Update the action value estimate with the incremental sample-average formula
        self.Q[action] = self.Q[action] + (1 / self.N[action]) * (reward - self.Q[action])


class GradientAgent:
    """
    Gradient bandit agent (policy gradient with softmax over preferences).
    Matches the API of your epsilon-greedy Agent: choose_action() and update_estimates(action, reward).

    - Maintains preferences H(a) instead of Q(a).
    - Samples actions from softmax(H).
    - Updates H using:   ΔH(a) ∝ (R_t - baseline) * (1{a=A_t} - π(a))
    - Optional running-average baseline reduces variance.

    Args:
        k (int): number of actions.
        alpha (float): step size for preference updates.
        use_baseline (bool): if True, use running average reward as baseline.
        init_prefs (float | array-like): initial value(s) for H. Default 0 for all actions.
    """

    def __init__(self, k, alpha=0.1, use_baseline=True, init_prefs=0.0):
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline

        # Preferences H and policy π
        self.H = np.full(k, init_prefs, dtype=float)
        self._update_policy()

        # For API parity / logging convenience
        self.Q = np.zeros(k)          # not used for action selection, but you can log it if you want
        self.N = np.zeros(k)          # count pulls per action
        self.action_rewards = np.zeros(k)

        # Baseline and timestep
        self.avg_reward = 0.0
        self.t = 0

    def _softmax(self, x):
        # numerically stable softmax
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.sum(e)

    def _update_policy(self):
        self.pi = self._softmax(self.H)

    # --------- public API ----------
    def choose_action(self) -> int:
        """
        Sample an action from the softmax policy over preferences.
        """
        return np.random.choice(self.k, p=self.pi)

    def update_estimates(self, action: int, reward: float):
        """
        Update preferences H using the gradient bandit rule (Eq. 2.12).
        Also maintains a running average reward as baseline if enabled.
        """
        # bookkeeping (optional; mirrors your epsilon agent)
        self.N[action] += 1
        self.action_rewards[action] += reward

        # update baseline
        self.t += 1
        if self.use_baseline:
            # incremental running average of rewards
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0.0

        # advantage signal
        advantage = reward - baseline

        # gradient ascent on preferences
        # H[a] += α * advantage * (1{a==action} - π[a])
        # vectorized form:
        one_hot = np.zeros(self.k)
        one_hot[action] = 1.0
        self.H += self.alpha * advantage * (one_hot - self.pi)

        # refresh policy
        self._update_policy()
