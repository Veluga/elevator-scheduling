from caller.caller import Call
from enum import Enum, auto
from config import NUM_FLOORS, NUM_ELEVATORS
from random import randint

class Environment:
    def __init__(self, floors=NUM_FLOORS, elevators=NUM_ELEVATORS, floor_height=4.0):
        self.floors = floors
        self.elevators = [Elevator() for _ in range(elevators)]
        self.up_calls = {i: False for i in range(floors)}
        self.down_calls = {i: False for i in range(floors)}
        self.floor_height = floor_height

    def call(self, floor, direction):
        if direction == Call.DOWN:
            self.down_calls[floor] = True
        else:
            self.up_calls[floor] = True

    def sample_state(self):
        # Generated calls
        up_calls = tuple(self.up_calls[i] for i in range(self.floors))
        down_calls = tuple(self.down_calls[i] for i in range(self.floors))
        # Elevator location, velocity, state, buttons pressed
        # Leave velocity out for simplification purposes
        elevators = tuple(
            tuple([e.cur_floor, e.state, tuple(e.buttons_pressed)]) 
            for e in self.elevators
        )
        return up_calls + down_calls + elevators

    def perform_action(self, action):
        def update_position(elevator, action):
            if action == ElevatorState.ASCENDING:
                if elevator.cur_floor == self.floors-1:
                    return -1
                else:
                    elevator.cur_floor += 1
            elif action == ElevatorState.DESCENDING:
                if elevator.cur_floor == 0:
                    return -1
                else:
                    elevator.cur_floor -= 1
            return 0

        def generate_destination_button_press(from_floor, direction):
            destination = randint(0, self.floors-1)
            if direction == Call.UP:
                while destination <= from_floor:
                    destination = randint(0, self.floors-1)
            else:
                while destination >= from_floor:
                    destination = randint(0, self.floors-1)
            return destination

        reward = 0
        for idx, elevator in enumerate(self.elevators):
            reward += update_position(elevator, action[idx])
            elevator.state = action[idx]
        for elevator in self.elevators:
            # Only stopped elevators can have passengers boarding or disembarking
            if elevator.state != ElevatorState.STOPPED:
                continue
            # Check whether ascending passengers boarding
            if self.up_calls[elevator.cur_floor]:
                self.up_calls[elevator.cur_floor] = False
                destination = generate_destination_button_press(
                    elevator.cur_floor, Call.UP
                )
                elevator.buttons_pressed.add(destination)
            # Check whether descending passengers boarding
            if self.down_calls[elevator.cur_floor]:
                self.down_calls[elevator.cur_floor] = False
                destination = generate_destination_button_press(
                    elevator.cur_floor, Call.DOWN
                )
                elevator.buttons_pressed.add(destination)
            # Check whether passengers disembarking
            if elevator.cur_floor in elevator.buttons_pressed:
                elevator.buttons_pressed.remove(elevator.cur_floor)
                reward += 1
        return reward
        

class Elevator:
    def __init__(self, cur_velocity=0.0, max_velocity=3.0, acceleration=1.0, cur_floor=0):
        self.cur_velocity = cur_velocity
        self.max_velocity = max_velocity
        self.acceleration = acceleration
        self.cur_floor = cur_floor
        self.interfloor_distance = 0.0
        self.state = ElevatorState.STOPPED
        self.buttons_pressed = set()
    
    def accelerate(self, delta_t):
        a = self.cur_velocity * delta_t
        a = max(0, a)
        a = min(self.max_velocity, a)
        self.cur_velocity = a

class ElevatorState(Enum):
    STOPPED = auto()
    ASCENDING = auto()
    DESCENDING = auto()