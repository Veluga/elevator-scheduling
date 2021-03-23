from building.building import ElevatorState
from .benchmark_agent import BenchmarkAgent
import settings as s

from copy import deepcopy
from collections import deque
from random import shuffle

class NearestCarScheduler(BenchmarkAgent):
    """Agent that implements a Nearest Car scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handling_elevator = 0

    def _best_suitability(self, elevators, floor, direction):
        """Calculate figure of suitability for each elevator.
        Return index of elevator with highest suitability.
        """
        elevator_indices = list(range(len(elevators)))
        # Shuffle indices to break ties randomly (i.e. prevent car 0 from being overused)
        shuffle(elevator_indices)

        best_fs = float('-inf')
        best_fs_idx = -1
        for idx in elevator_indices:
            elevator = elevators[idx]
            fs = float('-inf')
            # Check if elevator is moving in direction of call
            if (floor >= elevator.cur_floor and elevator.direction == ElevatorState.ASCENDING) \
            or (floor <= elevator.cur_floor and elevator.direction == ElevatorState.DESCENDING):
                # Check if call direction is along direction of elevator
                if direction == elevator.direction:
                    fs = s.NUM_FLOORS + 1 - abs(floor - elevator.cur_floor)
                else:
                    fs = s.NUM_FLOORS - abs(floor - elevator.cur_floor)
            else:
                fs = 1
            if fs > best_fs:
                    best_fs = fs
                    best_fs_idx = idx
        return best_fs_idx

    def assign_calls(self, new_up_calls, new_down_calls, elevators=None):
        for floor in new_up_calls:
            best_suitability_idx = self._best_suitability(elevators, floor, ElevatorState.ASCENDING)
            self.elevator_up_queues[best_suitability_idx].append(floor)
        for floor in new_down_calls:
            best_suitability_idx = self._best_suitability(elevators, floor, ElevatorState.DESCENDING)
            self.elevator_down_queues[best_suitability_idx].append(floor)

    def handle_unused(self, *args):
        # Transition to idle state
        return ElevatorState.STOPPED