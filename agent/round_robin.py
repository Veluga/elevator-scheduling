from building.building import ElevatorState
from .agent import Agent
import settings as s

class RoundRobinAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_up_calls = None
        self.prev_down_calls = None
        self.elevator_queues = [set() for _ in range(s.NUM_ELEVATORS)]
        self.handling_elevator = 0

    def _get_new_calls(self, up_calls, down_calls):
        print(self.prev_up_calls)
        print(self.prev_down_calls)
        print(up_calls)
        print(down_calls)
        new_calls = set()
        for f in range(s.NUM_FLOORS):
            new_calls |= (up_calls[f] | down_calls[f]) - (self.prev_up_calls[f] | self.prev_down_calls[f])
        return new_calls

    def _button_press_in_direction(self, elevator):
        if elevator.direction == ElevatorState.ASCENDING:
            return any(bp > elevator.cur_floor for bp in elevator.buttons_pressed)
        elif elevator.direction == ElevatorState.DESCENDING:
            return any(bp < elevator.cur_floor for bp in elevator.buttons_pressed)

    def get_action(self, state):
        if self.prev_up_calls is None:
            self.prev_up_calls = state['up_calls']
            self.prev_down_calls = state['down_calls']
            return [ElevatorState.STOPPED] * 4

        if state['up_calls'] != self.prev_up_calls or state['down_calls'] != self.prev_down_calls:
            new_calls = self._get_new_calls(state['up_calls'], state['down_calls'])
            for from_, _ in new_calls:
                self.elevator_queues[self.handling_elevator].add(from_)
                self.handling_elevator = (self.handling_elevator+1) % s.NUM_ELEVATORS
        
        actions = []
        for elevator in state['elevators']:
            if elevator.cur_floor in elevator.buttons_pressed:
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
            else:
                # Transition to idle state
                actions.append(ElevatorState.STOPPED)
        return actions

    def perform_update(self, state, action, reward, new_state):
        pass