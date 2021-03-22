from building.building import ElevatorState
from .benchmark_agent import BenchmarkAgent
from copy import deepcopy
from collections import deque
import settings as s

class RoundRobinAgent(BenchmarkAgent):
    """Agent that implements a FCFS Round Robin scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handling_elevator = 0

    def assign_calls(self, new_calls):
        for floor in new_calls:
            # Assign new call to elevator in round robin fashion
            self.elevator_queues[self.handling_elevator].append(floor)
            self.handling_elevator = (self.handling_elevator+1) % s.NUM_ELEVATORS

    def handle_unused(self, *args):
        # Transition to idle state
        return ElevatorState.STOPPED