from abc import ABC, abstractmethod
import settings as s

class Agent(ABC):
    def __init__(self, elevators=s.NUM_ELEVATORS):
        self.elevators = elevators

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def perform_update(self, state, action, reward, new_state):
        pass