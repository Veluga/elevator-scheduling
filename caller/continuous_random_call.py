import settings as s
from .caller import Call, Caller
from random import randint, random

class ContinuousRandomCallCaller(Caller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_call(self):
        floor = randint(0, self.floors-1)
        if floor == 0:
            return floor, Call.UP
        elif floor == self.floors-1:
            return floor, Call.DOWN
        else:
            return floor, Call.UP if random() > 0.5 else Call.DOWN