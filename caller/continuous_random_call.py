import settings as s
from .caller import Caller
from random import randint

class ContinuousRandomCallCaller(Caller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_call(self):
        if not self.call_eligible():
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