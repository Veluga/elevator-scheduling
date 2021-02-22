import settings as s
from abc import ABC, abstractmethod
from caller.caller import Call
from enum import IntEnum

class Building(ABC):
    def __init__(self, caller, floors=s.NUM_FLOORS, elevators=s.NUM_ELEVATORS, floor_height=s.FLOOR_HEIGHT):
        self.floors = floors
        self.caller = caller
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

    def reset(self):
        self.elevators = [Elevator() for _ in self.elevators]
        self._reset()

    @abstractmethod
    def _reset(self):
        pass

class ElevatorState(IntEnum):
    STOPPED = 0
    ASCENDING = 1
    DESCENDING = 2


class Elevator:
    def __init__(self, cur_floor=0, direction=ElevatorState.ASCENDING):
        self.cur_floor = cur_floor
        self.state = ElevatorState.STOPPED
        self.buttons_pressed = set()
        self.direction = direction