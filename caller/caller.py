import settings as s
from abc import ABC, abstractmethod
from enum import Enum, auto
from random import randint

class Call(Enum):
    UP = auto()
    DOWN = auto()

class Caller:
    def __init__(self, floors=s.NUM_FLOORS):
        self.floors = floors

    def call_eligible(self):
        return randint(0, s.AVERAGE_CALL_FREQUENCY) == s.AVERAGE_CALL_FREQUENCY

    @abstractmethod
    def generate_call(self):
        pass