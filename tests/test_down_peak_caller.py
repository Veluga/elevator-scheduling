from ..caller.down_peak_caller import DownPeakCaller
from ..settings import AVERAGE_CALL_FREQUENCY, NUM_FLOORS

import pytest

@pytest.fixture
def caller():
    caller = DownPeakCaller()
    yield caller

def test_destination_is_ground_floor(caller):
    for _ in range(100*AVERAGE_CALL_FREQUENCY*NUM_FLOORS):
        _, dest = caller.generate_call()
        if dest is not None:
            assert dest == 0