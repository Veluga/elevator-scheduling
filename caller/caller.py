from random import randint, random
from enum import Enum, auto
from config import NUM_FLOORS

class Call(Enum):
    UP = auto()
    DOWN = auto()

class Caller:
    def __init__(self, floors=NUM_FLOORS):
        self.floors = floors

    def generate_call(self):
        floor = randint(0, self.floors-1)
        if floor == 0:
            return floor, Call.UP
        elif floor == self.floors-1:
            return floor, Call.DOWN
        else:
            return floor, Call.UP if random() > 0.5 else Call.DOWN

