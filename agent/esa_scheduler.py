from building.building import ElevatorState
from .benchmark_agent import BenchmarkAgent
from copy import deepcopy
from collections import deque
import settings as s

class RoundRobinAgent(BenchmarkAgent):
    """Agent that implements an Empty-The-System scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assign_calls(self, new_calls):
        pass

    def handle_unused(self, *args):
        # Transition to idle state
        return ElevatorState.STOPPED