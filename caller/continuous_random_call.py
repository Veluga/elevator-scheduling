import settings as s
from .caller import Call, Caller
from random import randint, random

class ContinuousRandomCallCaller(Caller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_call(self):
        if randint(0, s.AVERAGE_CALL_FREQUENCY) < s.AVERAGE_CALL_FREQUENCY:
            return None, None

        call_floor = randint(0, self.floors-1)
        if call_floor == self.floors-1:
            destination_floor = randint(0, call_floor-1)
        elif call_floor == 0:
            destination_floor = randint(1, self.floors-1)
        else:
            destination_floor = randint(0, self.floors-1)
            while destination_floor == call_floor:
                destination_floor = randint(0, self.floors-1)
        return call_floor, destination_floor