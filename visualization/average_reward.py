import matplotlib.pyplot as plt
from .visualization import Visualization
from collections import deque

class AverageReward(Visualization):
    """Visualization of average reward over sliding window of last `sliding_window_size` timesteps."""
    def __init__(self, sliding_window_size=1000):
        self.sliding_window_size = sliding_window_size
        self.reward_sliding_window = deque(maxlen=self.sliding_window_size)
        self.average_rewards = []

    def next_reward(self, reward):
        if len(self.reward_sliding_window) < self.sliding_window_size:
            self.reward_sliding_window.append(sum(reward))
            self.average_rewards.append(sum(self.reward_sliding_window) / len(self.reward_sliding_window))
        else:
            # Instead of expensively calculating sum using `sum` (O(N)), perform O(1) arithmetic on last average_rewards value
            avg = self.average_rewards[-1] 
            avg -= self.reward_sliding_window[0] / self.sliding_window_size
            avg += sum(reward) / self.sliding_window_size
            self.reward_sliding_window.append(sum(reward))
            self.average_rewards.append(avg)

    def display(self):
        # Display plot using matplotlib
        plt.plot([i for i in range(len(self.average_rewards))], self.average_rewards, 'r-')
        plt.xlabel("Timestep")
        plt.ylabel("Avg. Reward, last " + str(len(self.reward_sliding_window)) + " timesteps")
        plt.show()