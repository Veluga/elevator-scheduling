import settings as s
from abc import ABC, abstractmethod
from caller.caller import Call
from enum import Enum, auto

class Building(ABC):
    def __init__(self, floors=s.NUM_FLOORS, elevators=s.NUM_ELEVATORS, floor_height=s.FLOOR_HEIGHT):
        self.floors = floors
        self.elevators = [Elevator() for _ in range(elevators)]
        self.floor_height = floor_height

    @abstractmethod
    def call(self, call_floor, destination_floor):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def perform_action(self, action):
        pass
        

class Elevator:
    def __init__(self, cur_velocity=0.0, cur_floor=0, max_velocity=s.MAX_VELOCITY, acceleration=s.ACCELERATION, floor_height=s.FLOOR_HEIGHT):
        self.cur_velocity = cur_velocity
        self.cur_floor = cur_floor
        self.max_velocity = max_velocity
        self.acceleration = acceleration
        self.floor_height = floor_height
        self.interfloor_distance = 0.0
        self.stopped = True
        self.direction = ElevatorState.ASCENDING
        self.buttons_pressed = set()

class ElevatorState(Enum):
    STOPPED = auto()
    ASCENDING = auto()
    DESCENDING = auto()