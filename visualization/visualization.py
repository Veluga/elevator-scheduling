from abc import ABC, abstractmethod

class Visualization(ABC):
    @abstractmethod
    def next_reward(self, reward):
        pass

    @abstractmethod
    def display(self):
        pass