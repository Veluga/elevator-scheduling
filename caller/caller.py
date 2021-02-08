import settings as s
from abc import ABC, abstractmethod
from enum import Enum, auto

class Call(Enum):
    UP = auto()
    DOWN = auto()

class Caller:
    def __init__(self, floors=s.NUM_FLOORS):
        self.floors = floors

    @abstractmethod
    def generate_call(self):
        pass