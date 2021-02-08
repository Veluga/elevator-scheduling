import settings as s
from abc import ABC, abstractmethod
from caller.caller import Call
from enum import Enum, auto

class Building(ABC):
    def __init__(self, floors=s.NUM_FLOORS, elevators=s.NUM_ELEVATORS, floor_height=s.FLOOR_HEIGHT):
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

    @abstractmethod
    def sample_state(self):
        pass

    def perform_action(self, action):
        pass
        

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