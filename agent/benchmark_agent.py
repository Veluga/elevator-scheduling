from .agent import Agent
from building.building import ElevatorState
import settings as s

from collections import deque
from copy import deepcopy
from abc import ABC, abstractmethod


class BenchmarkAgent(Agent, ABC):
    """Agent that implements a zoning strategy whereby every car serves only a set of neighbouring floors.
    Zones are determined statically upon initialization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_up_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.prev_down_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.elevator_queues = [deque() for _ in range(s.NUM_ELEVATORS)]

    def _get_new_calls(self, up_calls, down_calls):
        """Compare current state of calls against previous state to detect new calls"""
        new_calls = set()
        for f in range(s.NUM_FLOORS):
            if up_calls[f] - self.prev_up_calls[f] or down_calls[f] - self.prev_down_calls[f]:
                new_calls.add(f)
        return new_calls

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
    def assign_calls(self, new_calls):
        pass

    @abstractmethod
    def handle_unused(self, idx, elevator):
        pass

    def get_action(self, state):
        new_calls = self._get_new_calls(state['up_calls'], state['down_calls'])
        self.assign_calls(new_calls)

        # Preserve current state
        self.prev_up_calls = deepcopy(state['up_calls'])
        self.prev_down_calls = deepcopy(state['down_calls'])

        actions = []
        for idx, elevator in enumerate(state['elevators']):
            if elevator.cur_floor in self.elevator_queues[idx]:
                # Must let passenger board
                actions.append(ElevatorState.STOPPED)
                self.elevator_queues[idx].remove(elevator.cur_floor)
                self.prev_up_calls[elevator.cur_floor] = set()
                self.prev_down_calls[elevator.cur_floor] = set()
            elif elevator.cur_floor in elevator.buttons_pressed:
                # Passengers must be allowed to disembark
                actions.append(ElevatorState.STOPPED)
            elif self._button_press_in_direction(elevator):
                # Continue travelling in direction
                actions.append(elevator.direction)
            elif len(elevator.buttons_pressed) > 0:
                # Invert direction to serve boarded passengers before picking up new passengers (FCFS)
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.direction == ElevatorState.DESCENDING
                    else ElevatorState.DESCENDING
                )
            elif len(self.elevator_queues[idx]) > 0:
                # Pick up passengers waiting in hall, serve call
                pickup_floor = self.elevator_queues[idx][0]
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.cur_floor < pickup_floor
                    else ElevatorState.DESCENDING
                )
            else:
                # No boarded passengers, no passengers in queue
                action = self.handle_unused(idx, elevator)
                actions.append(action)
        return actions

    def perform_update(self, state, action, reward, new_state):
        """Unused for benchmark agent that doesn't learn from experience."""
        pass