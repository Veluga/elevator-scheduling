#import settings as s
from abc import ABC, abstractmethod

class Agent(ABC):
    #def __init__(self, elevators=s.NUM_ELEVATORS):
    def __init__(self, elevators=3):
        self.elevators = elevators

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def perform_update(self, state, action, reward, new_state):
        pass