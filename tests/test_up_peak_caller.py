from ..caller.up_peak_caller import UpPeakCaller
from ..settings import AVERAGE_CALL_FREQUENCY, NUM_FLOORS

import pytest

@pytest.fixture
def caller():
    caller = UpPeakCaller()
    yield caller

def test_source_is_ground_floor(caller):
    for _ in range(100*AVERAGE_CALL_FREQUENCY*NUM_FLOORS):
        src, _ = caller.generate_call()
        if src is not None:
            assert src == 0