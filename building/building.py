import settings as s
from abc import ABC, abstractmethod
from caller.caller import Call
from enum import IntEnum


class Building(ABC):
    """Abstract building class that controllers can program against."""
    def __init__(self, caller, floors=s.NUM_FLOORS, elevators=s.NUM_ELEVATORS):
        self.floors = floors
        self.caller = caller
        self.elevators = [Elevator() for _ in range(elevators)]

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
    """Enum used to capture elevator state and direction of travel."""
    STOPPED = 0
    ASCENDING = 1
    DESCENDING = 2


class Elevator:
    """Dataclass that holds information about a single elevator."""
    def __init__(self, cur_floor=0, direction=ElevatorState.ASCENDING):
        self.cur_floor = cur_floor
        self.state = ElevatorState.STOPPED
        self.buttons_pressed = set()
        self.direction = direction