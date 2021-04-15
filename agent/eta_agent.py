from building.building import ElevatorState
from .benchmark_agent import BenchmarkAgent
from copy import deepcopy
from collections import deque
import settings as s

class ETAAgent(BenchmarkAgent):
    """Agent that implements a FCFS Round Robin scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queued_calls = []

    def _travelling_towards(self, floor, elevator):
        return (elevator.direction == ElevatorState.ASCENDING and floor >= elevator.cur_floor) \
        or (elevator.direction == ElevatorState.DESCENDING and floor <= elevator.cur_floor)

    def _available_elevators(self, call_floor, call_direction, elevators):
        available_elevators = []
        for idx, elevator in enumerate(elevators):
            # Elevator moves towards call in direction of call or is idle
            if ((call_direction == elevator.direction or call_floor == 0 or call_floor == s.NUM_FLOORS-1) \
            and self._travelling_towards(call_floor, elevator)) \
            or  (elevator.empty() and len(self.elevator_up_queues[idx]) == 0 and len(self.elevator_down_queues[idx]) == 0):
                available_elevators.append((idx, elevator))
        return available_elevators

    def _estimated_travel_cost(self, call_floor, elevator_idx, elevator):
        stops_between = 0
        if elevator.cur_floor <= call_floor:
            for f in range(elevator.cur_floor, call_floor):
                stops_between += f in elevator.buttons_pressed
                stops_between += f in self.elevator_up_queues[elevator_idx]
        else:
            for f in range(call_floor, elevator.cur_floor):
                stops_between += f in elevator.buttons_pressed
                stops_between += f in self.elevator_up_queues[elevator_idx]
        return stops_between+1

    def _min_cost_elevator(self, call_floor, elevators):
        min_cost = float('inf')
        min_cost_elevator = None
        for idx, e in elevators:
            estimated_travel_cost = self._estimated_travel_cost(call_floor, idx, e)
            if estimated_travel_cost < min_cost:
                min_cost = estimated_travel_cost
                min_cost_elevator = idx
        return min_cost_elevator

    def assign_calls(self, new_up_calls, new_down_calls, elevators):
        unserved_calls = [
            (f, ElevatorState.ASCENDING) for f in new_up_calls
        ] + [
            (f, ElevatorState.DESCENDING) for f in new_down_calls
        ] + self.queued_calls
        self.queued_calls = []

        for floor, direction in unserved_calls:
            available_elevators = self._available_elevators(floor, direction, elevators)
            if available_elevators == []:
                self.queued_calls.append((floor, direction))
                continue
            min_cost_idx = self._min_cost_elevator(floor, available_elevators)
            if direction == ElevatorState.ASCENDING:
                self.elevator_up_queues[min_cost_idx].append(floor)
            else:
                self.elevator_down_queues[min_cost_idx].append(floor)

    def handle_unused(self, *args):
        # Transition to idle state
        return ElevatorState.STOPPED