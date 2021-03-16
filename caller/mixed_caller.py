import settings as s
from .caller import Caller
from .down_peak_caller import DownPeakCaller
from .up_peak_caller import UpPeakCaller
from .interfloor_caller import InterfloorCaller
from random import randint, random

class MixedCaller(Caller):
    def __init__(self, up_peak_prob=1/3, down_peak_prob=1/3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up_peak_prob = up_peak_prob
        self.down_peak_prob = down_peak_prob
        self.up_peak_caller = UpPeakCaller()
        self.down_peak_caller = DownPeakCaller()
        self.interfloor_caller = InterfloorCaller()
        self.up_peak_caller.call_eligible = lambda *args, **kwargs: True
        self.down_peak_caller.call_eligible = lambda *args, **kwargs: True
        self.interfloor_caller.call_eligible = lambda *args, **kwargs: True

    def generate_call(self):
        if not self.call_eligible():
            return None, None

        roll = random()
        if roll <= self.down_peak_prob:
            return self.down_peak_caller.generate_call()
        elif roll <= self.down_peak_prob + self.up_peak_prob:
            return self.up_peak_caller.generate_call()
        else:
            return self.interfloor_caller.generate_call()