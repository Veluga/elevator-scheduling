import matplotlib.pyplot as plt
from .visualization import Visualization

class CumulativeReward(Visualization):
    def __init__(self):
        self.cumulative_reward = 0
        self.cumulative_rewards = []

    def next_reward(self, reward):
        self.cumulative_reward += reward
        self.cumulative_rewards.append(self.cumulative_reward)

    def display(self):
        plt.plot([i for i in range(len(self.cumulative_rewards))], self.cumulative_rewards, 'r-')
        plt.show()