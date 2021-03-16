import settings as s
from abc import ABC, abstractmethod
from enum import Enum, auto
from random import randint

class Call(Enum):
    """Types of calls."""
    UP = auto()
    DOWN = auto()

class Caller:
    """Abstract caller class that controllers can program against."""
    def __init__(self, floors=s.NUM_FLOORS):
        self.floors = floors

    def call_eligible(self):
        # An AVERAGE_CALL_FREQUENCY of n means that a call should be generated, on average, every n timesteps
        return randint(1, s.AVERAGE_CALL_FREQUENCY) == s.AVERAGE_CALL_FREQUENCY

    @abstractmethod
    def generate_call(self):
        pass