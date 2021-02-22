from building.building import ElevatorState
from .agent import Agent
from copy import deepcopy
from collections import deque
import settings as s

class RoundRobinAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_up_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.prev_down_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.elevator_queues = [deque() for _ in range(s.NUM_ELEVATORS)]
        self.handling_elevator = 0

    def _get_new_calls(self, up_calls, down_calls):
        new_calls = set()
        for f in range(s.NUM_FLOORS):
            if up_calls[f] - self.prev_up_calls[f] or down_calls[f] - self.prev_down_calls[f]:
                new_calls.add(f)
        return new_calls

    def _button_press_in_direction(self, elevator):
        if elevator.direction == ElevatorState.ASCENDING:
            return any(bp > elevator.cur_floor for bp in elevator.buttons_pressed)
        elif elevator.direction == ElevatorState.DESCENDING:
            return any(bp < elevator.cur_floor for bp in elevator.buttons_pressed)

    def get_action(self, state):
        new_calls = self._get_new_calls(state['up_calls'], state['down_calls'])
        for floor in new_calls:
            self.elevator_queues[self.handling_elevator].append(floor)
            self.handling_elevator = (self.handling_elevator+1) % s.NUM_ELEVATORS
        
        self.prev_up_calls = deepcopy(state['up_calls'])
        self.prev_down_calls = deepcopy(state['down_calls'])

        actions = []
        for idx, elevator in enumerate(state['elevators']):
            if elevator.cur_floor in self.elevator_queues[idx]:
                # Let passenger board
                actions.append(ElevatorState.STOPPED)
                self.elevator_queues[idx].remove(elevator.cur_floor)
                self.prev_up_calls[elevator.cur_floor] = set()
                self.prev_down_calls[elevator.cur_floor] = set()
            elif elevator.cur_floor in elevator.buttons_pressed:
                # Unload passenger
                actions.append(ElevatorState.STOPPED)
            elif self._button_press_in_direction(elevator):
                # Continue travelling in direction
                actions.append(elevator.direction)
            elif len(elevator.buttons_pressed) > 0:
                # Invert direction
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.direction == ElevatorState.DESCENDING
                    else ElevatorState.DESCENDING
                )
            elif len(self.elevator_queues[idx]) > 0:
                # Pick up passenger for call
                pickup_floor = self.elevator_queues[idx][0]
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.cur_floor < pickup_floor
                    else ElevatorState.DESCENDING
                )
            else:
                # Transition to idle state
                actions.append(ElevatorState.STOPPED)
        
        return actions

    def perform_update(self, state, action, reward, new_state):
        pass