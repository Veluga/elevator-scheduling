import settings as s
from abc import ABC, abstractmethod
from caller.caller import Call
from enum import IntEnum
import uuid

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
    """Class that holds information about a single elevator."""
    def __init__(self, cur_floor=0, direction=ElevatorState.ASCENDING, max_capacity=s.ELEVATOR_MAX_CAPACITY):
        self.cur_floor = cur_floor
        self.state = ElevatorState.STOPPED
        self.buttons_pressed = set()
        self.direction = direction
        self.passengers = set()
        self.max_capacity = max_capacity

    def empty(self):
        return len(self.buttons_pressed) == 0

    def full(self):
        return len(self.passengers) >= self.max_capacity

class Passenger:
    """Class that holds information about a single passenger."""
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.system_time = 0
        self.waiting_time = 0
        self.in_elevator = False
        self.served = False

    def wait_tick(self):
        if not self.served:
            self.system_time += s.TICK_LENGTH_IN_SECONDS
            if not self.in_elevator:
                self.waiting_time += s.TICK_LENGTH_IN_SECONDS
    
    def mark_served(self):
        self.served = True
        self.in_elevator = False

    def enter_elevator(self):
        self.in_elevator = True