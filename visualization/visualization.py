from abc import ABC, abstractmethod

class Visualization(ABC):
    """Abstract visualization class that controllers can program against."""
    @abstractmethod
    def next_reward(self, reward):
        pass

    @abstractmethod
    def display(self):
        pass