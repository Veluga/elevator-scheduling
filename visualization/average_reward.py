import matplotlib.pyplot as plt
from .visualization import Visualization
from collections import deque

class AverageReward(Visualization):
    def __init__(self, sliding_window_size=1000):
        self.reward_sliding_window = deque(maxlen=sliding_window_size)
        self.average_rewards = []

    def next_reward(self, reward):
        self.reward_sliding_window.append(reward)
        self.average_rewards.append(sum(self.reward_sliding_window))

    def display(self):
        plt.plot([i for i in range(len(self.average_rewards))], self.average_rewards, 'r-')
        plt.xlabel("Timestep")
        plt.ylabel("Avg. Reward, last " + str(len(self.reward_sliding_window)) + " timesteps")
        plt.show()