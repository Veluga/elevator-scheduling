import matplotlib.pyplot as plt
from .visualization import Visualization

class CumulativeReward(Visualization):
    """Visualization of cumulative reward over all previous timesteps."""
    def __init__(self):
        self.cumulative_reward = 0
        self.cumulative_rewards = []

    def next_reward(self, reward):
        self.cumulative_reward += sum(reward)
        self.cumulative_rewards.append(self.cumulative_reward)

    def display(self):
        # Plot using matplotlib
        #plt.plot(range(len(self.cumulative_rewards)), self.cumulative_rewards, 'r-')
        #plt.show()
        print(self.cumulative_reward)