import settings as s
from .building import Building, Call, Elevator, ElevatorState
from random import randint
import numpy as np

class DiscreteFloorTransition(Building):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def call(self, call_floor, destination_floor):
        if call_floor < destination_floor:
            self.up_calls[call_floor].add(destination_floor)
        else:
            self.down_calls[call_floor].add(destination_floor)

    def sample_state(self):
        np_up_calls = np.array([[len(s) > 0 for f, s in self.up_calls.items()]], dtype=int)
        np_down_calls = np.array([[len(s) > 0 for f, s in self.down_calls.items()]], dtype=int)
        np_state_vector = np.append(np_up_calls, np_down_calls)
        for e in self.elevators:
            np_state_vector = np.append(np_state_vector, [e.cur_floor])
            np_state_vector = np.append(np_state_vector, [s == e.state for s in range(min(ElevatorState).value, max(ElevatorState).value+1)])
            np_state_vector = np.append(np_state_vector, [f in e.buttons_pressed for f in range(self.floors)])
        return np_state_vector, {'up_calls': self.up_calls, 'down_calls': self.down_calls, 'elevators': self.elevators}

    def perform_action(self, actions):
        def update_position(elevator, action):
            if action == ElevatorState.ASCENDING:
                if elevator.cur_floor != self.floors-1:
                    elevator.cur_floor += 1
                    elevator.direction = ElevatorState.ASCENDING
            elif action == ElevatorState.DESCENDING:
                if elevator.cur_floor != 0:
                    elevator.cur_floor -= 1
                    elevator.direction = ElevatorState.DESCENDING
            return 0
        
        from_, to = self.caller.generate_call()
        if from_ is not None and to is not None:
            self.call(from_, to)

        rewards = []
        for elevator, action in zip(self.elevators, actions):
            rewards.append(0)
            rewards[-1] += update_position(elevator, action)
            elevator.state = action
            
            # Only stopped elevators can have passengers boarding or disembarking
            if elevator.state != ElevatorState.STOPPED:
                continue
            
            # Check whether ascending passengers boarding
            if len(self.up_calls[elevator.cur_floor]) > 0:
                for destination in self.up_calls[elevator.cur_floor]:
                    elevator.buttons_pressed.add(destination)
                self.up_calls[elevator.cur_floor] = set()

            # Check whether descending passengers boarding
            if len(self.down_calls[elevator.cur_floor]) > 0:
                for destination in self.down_calls[elevator.cur_floor]:
                    elevator.buttons_pressed.add(destination)
                self.down_calls[elevator.cur_floor] = set()

            # Check whether passengers disembarking
            if elevator.cur_floor in elevator.buttons_pressed:
                rewards[-1] += s.REWARD_DELIVERED_PASSENGER
                elevator.buttons_pressed.remove(elevator.cur_floor)
        return rewards
    
    def _reset(self):
        self.up_calls = {floor_num: set() for floor_num in range(self.floors)}
        self.down_calls = {floor_num: set() for floor_num in range(self.floors)}