from .down_peak_caller import DownPeakCaller
from .up_peak_caller import UpPeakCaller
from .interfloor_caller import InterfloorCaller
from .mixed_caller import MixedCaller
import settings as s


def get_caller():
    if s.TRAFFIC_PROFILE == "mixed":
        return MixedCaller()
    elif s.TRAFFIC_PROFILE == "interfloor":
        return InterfloorCaller()
    elif s.TRAFFIC_PROFILE == "up-peak":
        return UpPeakCaller()
    elif s.TRAFFIC_PROFILE == "down-peak":
        return DownPeakCaller()
    raise ValueError(
        f"Unknown traffic pattern specified in settings: {s.TRAFFIC_PROFILE}\nMust be one of mixed/interfloor/up-peak/down-peak."
    )