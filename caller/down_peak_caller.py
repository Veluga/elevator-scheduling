import settings as s
from .caller import Caller
from random import randint

class DownPeakCaller(Caller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_call(self):
        if not self.call_eligible():
            return None, None

        call_floor = randint(1, self.floors-1)
        return call_floor, 0