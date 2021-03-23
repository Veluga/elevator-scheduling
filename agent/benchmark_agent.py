from .agent import Agent
from building.building import ElevatorState
import settings as s

from collections import deque
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Call:
    """Stores information about calls that is relevant for schedulers."""
    floor: int 
    t_wait: int

class ElevatorQueue:
    """Thin wrapper of deque to make code in `get_action` of `BenchmarkAgent` cleaner."""
    def __init__(self):
        self.q = deque()
    
    def append(self, floor):
        self.q.append(Call(floor, 0))

    def increase_waiting_times(self):
        for call in self.q:
            call.t_wait += s.TICK_LENGTH_IN_SECONDS
    
    def remove(self, floor):
        to_remove = []
        for call in self.q:
            if call.floor == floor:
                to_remove.append(call)
        for call in to_remove:
            self.q.remove(call)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, key):
        return self.q[key]

    def __contains__(self, floor):
        for call in self.q:
            if call.floor == floor:
                return True
        return False

class BenchmarkAgent(Agent, ABC):
    """Agent that implements a zoning strategy whereby every car serves only a set of neighbouring floors.
    Zones are determined statically upon initialization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_up_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.prev_down_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        
        self.elevator_up_queues = [ElevatorQueue() for _ in range(s.NUM_ELEVATORS)]
        self.elevator_down_queues = [ElevatorQueue() for _ in range(s.NUM_ELEVATORS)]

    def _get_new_calls(self, up_calls, down_calls):
        """Compare current state of calls against previous state to detect new calls"""
        new_up_calls = set()
        new_down_calls = set()
        for floor in range(s.NUM_FLOORS):
            if up_calls[floor] - self.prev_up_calls[floor] != set():
                new_up_calls.add(floor)
            if down_calls[floor] - self.prev_down_calls[floor] != set():
                new_down_calls.add(floor)
        return new_up_calls, new_down_calls

    def _button_press_in_direction(self, elevator):
        """Determine whether any button press in direction of travel remains.
        An elevator must continue travelling in its current direction if there are
        unserved button presses in the direction of travel (i.e. it cannot reverse to pick up
        passengers in the opposite direction).
        """
        if elevator.direction == ElevatorState.ASCENDING:
            return any(bp > elevator.cur_floor for bp in elevator.buttons_pressed)
        elif elevator.direction == ElevatorState.DESCENDING:
            return any(bp < elevator.cur_floor for bp in elevator.buttons_pressed)

    @abstractmethod
    def assign_calls(self, new_up_calls, new_down_calls):
        """Derived classes must decide how to delegate calls to elevators."""
        pass

    @abstractmethod
    def handle_unused(self, idx, elevator):
        """Derived classes must decide what unused (no boarded/waiting passengers) elevators should do."""
        pass
    
    def handle_waiting(self, idx, elevator):
        # Pick up passengers waiting in hall, serving oldest call (FCFS)
        oldest_call_up = self.elevator_up_queues[idx][0] if len(self.elevator_up_queues[idx]) > 0 else Call(0, float('-inf'))
        oldest_call_down = self.elevator_down_queues[idx][0] if len(self.elevator_down_queues[idx]) > 0 else Call(0, float('-inf'))
        if elevator.cur_floor < (oldest_call_up.floor if oldest_call_up.t_wait > oldest_call_down.t_wait else oldest_call_down.floor):
            return ElevatorState.ASCENDING
        else:
            return ElevatorState.DESCENDING

    def get_action(self, state):
        """Template method that handles common elevator logic (e.g. elevators must continue in direction of
        travel until all button presses in direction have been served).
        Delegates benchmark specific behaviour via `assign_calls` and `handle_unused` methods.
        Called in loop of `benchmark_controller`.
        """
        # Increase waiting time of all passengers in hall (i.e. unserved calls)
        for q in self.elevator_up_queues + self.elevator_down_queues:
            q.increase_waiting_times()

        # Deduce newly generated calls from previous/current state comparison
        new_up_calls, new_down_calls = self._get_new_calls(state['up_calls'], state['down_calls'])
        self.assign_calls(new_up_calls, new_down_calls, state['elevators'])

        # Preserve current state
        self.prev_up_calls = deepcopy(state['up_calls'])
        self.prev_down_calls = deepcopy(state['down_calls'])

        actions = []
        for idx, elevator in enumerate(state['elevators']):
            if (elevator.direction == ElevatorState.ASCENDING or elevator.cur_floor == 0 or elevator.empty()) \
            and elevator.cur_floor in self.elevator_up_queues[idx]:
                # Must let ascending passenger board
                actions.append(ElevatorState.STOPPED)
                self.elevator_up_queues[idx].remove(elevator.cur_floor)
                self.prev_up_calls[elevator.cur_floor] = set()
            elif (elevator.direction == ElevatorState.DESCENDING or elevator.cur_floor == s.NUM_FLOORS-1 or elevator.empty()) \
            and elevator.cur_floor in self.elevator_down_queues[idx]:
                # Must let descending passenger board
                actions.append(ElevatorState.STOPPED)
                self.elevator_down_queues[idx].remove(elevator.cur_floor)
                self.prev_down_calls[elevator.cur_floor] = set()
            elif elevator.cur_floor in elevator.buttons_pressed:
                # Passengers must be allowed to disembark
                actions.append(ElevatorState.STOPPED)
            elif self._button_press_in_direction(elevator):
                # Continue travelling in direction
                actions.append(elevator.direction)
            elif len(elevator.buttons_pressed) > 0:
                # Invert direction to serve boarded passengers before picking up new passengers
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.direction == ElevatorState.DESCENDING
                    else ElevatorState.DESCENDING
                )
            elif len(self.elevator_up_queues[idx]) > 0 or len(self.elevator_down_queues[idx]) > 0:
                action = self.handle_waiting(idx, elevator)
                actions.append(action) 
            else:
                # No boarded passengers, no passengers in queue
                action = self.handle_unused(idx, elevator)
                actions.append(action)
        return actions

    def perform_update(self, state, action, reward, new_state):
        """Unused for benchmark agent that doesn't learn from experience."""
        pass