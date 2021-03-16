import settings as s
from .caller import Caller
from .down_peak_caller import DownPeakCaller
from .up_peak_caller import UpPeakCaller
from .interfloor_caller import InterfloorCaller
from random import randint, random

class MixedCaller(Caller):
    """Caller that implements a mixed traffic profile (e.g. observed in an office during lunch time).
    A call will be an up-peak call with probability `up_peak_prob`, a down-peak call with probability
    `down_peak_prob`, and an interfloor call with probability `1 - up_peak_prob - down_peak_prob`.
    """
    def __init__(self, up_peak_prob=1/3, down_peak_prob=1/3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up_peak_prob = up_peak_prob
        self.down_peak_prob = down_peak_prob
        self.up_peak_caller = UpPeakCaller()
        self.down_peak_caller = DownPeakCaller()
        self.interfloor_caller = InterfloorCaller()
        # Replace call_eligible function with function that always returns True since
        # eligibility is already handled by MixedCaller instance.
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