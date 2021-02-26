import settings as s
from .building import Building, Call, Elevator, ElevatorState
from random import randint
from copy import deepcopy
import numpy as np

class DiscreteFloorTransition(Building):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def call(self, call_floor, destination_floor):
        if call_floor < destination_floor:
            self.up_calls[call_floor].add(destination_floor)
            self.up_call_waiting_times[call_floor] = max(self.up_call_waiting_times[call_floor], 1)
        else:
            self.down_calls[call_floor].add(destination_floor)
            self.down_call_waiting_times[call_floor] = max(self.down_call_waiting_times[call_floor], 1)

    def sample_state(self):
        state_dict = {
            'up_calls': deepcopy(self.up_calls),
            'down_calls': deepcopy(self.down_calls),
            'elevators': deepcopy(self.elevators),
            'up_call_waiting_times': deepcopy(self.up_call_waiting_times),
            'down_call_waiting_times': deepcopy(self.down_call_waiting_times)
        }
        np_up_calls = np.array([[len(s) > 0 for f, s in self.up_calls.items()]], dtype=int)
        np_down_calls = np.array([[len(s) > 0 for f, s in self.down_calls.items()]], dtype=int)
        np_state_vector = np.append(np_up_calls, np_down_calls)
        np_state_vector = np.append(np_state_vector, [self.up_call_waiting_times[f] for f in range(self.floors)])
        np_state_vector = np.append(np_state_vector, [self.down_call_waiting_times[f] for f in range(self.floors)])
        for e in self.elevators:
            np_state_vector = np.append(np_state_vector, [e.cur_floor])
            np_state_vector = np.append(np_state_vector, [s == e.state for s in range(min(ElevatorState).value, max(ElevatorState).value+1)])
            np_state_vector = np.append(np_state_vector, [f in e.buttons_pressed for f in range(self.floors)])
            np_state_vector = np.append(np_state_vector, [e.button_press_waiting_times[f] for f in range(self.floors)])
        return np_state_vector, state_dict

    def increment_waiting_times(self):
        for f in range(self.floors):
            if self.up_call_waiting_times[f] > 0:
                self.up_call_waiting_times[f] += s.TICK_LENGTH_IN_SECONDS
            if self.down_call_waiting_times[f] > 0:
                self.down_call_waiting_times[f] += s.TICK_LENGTH_IN_SECONDS
            for e in self.elevators:
                if e.button_press_waiting_times[f] > 0:
                    e.button_press_waiting_times[f] += s.TICK_LENGTH_IN_SECONDS

    def calculate_waiting_times(self, op=lambda x: x):
        total_waiting_time = 0
        for f in range(self.floors):
            total_waiting_time -= op(self.up_call_waiting_times[f])
            total_waiting_time -= op(self.down_call_waiting_times[f])
            for e in self.elevators:
                total_waiting_time -= op(e.button_press_waiting_times[f])
        return total_waiting_time

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
                    elevator.button_press_waiting_times[destination] = max(
                        elevator.button_press_waiting_times[destination],
                        self.up_call_waiting_times[elevator.cur_floor]
                    )
                self.up_calls[elevator.cur_floor] = set()
                self.up_call_waiting_times[elevator.cur_floor] = 0

            # Check whether descending passengers boarding
            if len(self.down_calls[elevator.cur_floor]) > 0:
                for destination in self.down_calls[elevator.cur_floor]:
                    elevator.buttons_pressed.add(destination)
                    elevator.button_press_waiting_times[destination] = max(
                        elevator.button_press_waiting_times[destination],
                        self.down_call_waiting_times[elevator.cur_floor]
                    )
                self.down_calls[elevator.cur_floor] = set()
                self.down_call_waiting_times[elevator.cur_floor] = 0

            # Check whether passengers disembarking
            if elevator.cur_floor in elevator.buttons_pressed:
                elevator.buttons_pressed.remove(elevator.cur_floor)
                elevator.button_press_waiting_times[elevator.cur_floor] = 0

        t_wait_total = self.calculate_waiting_times()
        self.increment_waiting_times()
        return t_wait_total
    
    def _reset(self):
        self.up_calls = {floor_num: set() for floor_num in range(self.floors)}
        self.down_calls = {floor_num: set() for floor_num in range(self.floors)}
        self.up_call_waiting_times = {f: 0 for f in range(self.floors)}
        self.down_call_waiting_times = {f: 0 for f in range(self.floors)}