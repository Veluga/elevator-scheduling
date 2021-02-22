import settings as s
from .caller import Caller
from random import randint

class UpPeakCaller(Caller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_call(self):
        if not self.call_eligible():
            return None, None

        destination_floor = randint(1, self.floors-1)
        return 0, destination_floor