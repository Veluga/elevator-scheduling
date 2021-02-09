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
    def __init__(self, cur_floor=0):
        self.cur_floor = cur_floor
        self.state = ElevatorState.STOPPED
        self.buttons_pressed = set()

class ElevatorState(Enum):
    STOPPED = auto()
    ASCENDING = auto()
    DESCENDING = auto()