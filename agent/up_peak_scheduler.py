from building.building import ElevatorState
from .benchmark_agent import BenchmarkAgent
from copy import deepcopy
from collections import deque
import settings as s

class UpPeakScheduler(BenchmarkAgent):
    """Agent that implements a round robin scheduler that returns cars to ground floor when unused."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handling_elevator = 0

    def assign_calls(self, new_up_calls, new_down_calls, elevators=None):
        for floor in new_up_calls:
            # Assign new call to elevator in round robin fashion
            self.elevator_up_queues[self.handling_elevator].append(floor)
            self.handling_elevator = (self.handling_elevator+1) % s.NUM_ELEVATORS
        for floor in new_down_calls:
            # Assign new call to elevator in round robin fashion
            self.elevator_down_queues[self.handling_elevator].append(floor)
            self.handling_elevator = (self.handling_elevator+1) % s.NUM_ELEVATORS

    def handle_unused(self, _, elevator):
        # Return to ground floor
        if elevator.cur_floor == 0:
            return ElevatorState.STOPPED
        else:
            return ElevatorState.DESCENDING