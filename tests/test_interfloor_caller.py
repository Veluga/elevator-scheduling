from ..caller.interfloor_caller import InterfloorCaller
from ..settings import AVERAGE_CALL_FREQUENCY, NUM_FLOORS

import pytest

@pytest.fixture
def caller():
    caller = InterfloorCaller()
    yield caller

def test_source_is_ground_floor(caller):
    for _ in range(100*AVERAGE_CALL_FREQUENCY*NUM_FLOORS):
        src, dest = caller.generate_call()
        if src is None:
            assert dest is None
        elif src is not None:
            assert src != dest
            if src == 0:
                assert src < dest
            elif src == NUM_FLOORS-1:
                assert src > dest