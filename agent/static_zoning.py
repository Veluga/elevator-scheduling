from .agent import Agent
from building.building import ElevatorState
import settings as s

from collections import deque
from copy import deepcopy

class StaticZoningAgent(Agent):
    """Agent that implements a zoning strategy whereby every car serves only a set of neighbouring floors.
    Zones are determined statically upon initialization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_up_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.prev_down_calls = {floor_num: set() for floor_num in range(s.NUM_FLOORS)}
        self.elevator_queues = [deque() for _ in range(s.NUM_ELEVATORS)]
        self.zones = self._calculate_zones()

    def _get_new_calls(self, up_calls, down_calls):
        """Compare current state of calls against previous state to detect new calls"""
        new_calls = set()
        for f in range(s.NUM_FLOORS):
            if up_calls[f] - self.prev_up_calls[f] or down_calls[f] - self.prev_down_calls[f]:
                new_calls.add(f)
        return new_calls

    def _calculate_zones(self, num_floors=s.NUM_FLOORS, num_elevators=s.NUM_ELEVATORS):
        serving_floors = []
        from_ = 0
        for i in range(num_elevators):
            floors_to_serve = int(num_floors / (num_elevators-i))
            serving_floors.append((from_, from_ + floors_to_serve))
            from_ += floors_to_serve
            num_floors -= floors_to_serve
        return serving_floors

    def _button_press_in_direction(self, elevator):
        """Determine whether any button press in direction of travel remains.
        An elevator must continue travelling in its current direction if there are
        unserved button presses in the direction of travel (i.e. it cannot reverse to pick up
        passengers in the opposite direction).
        """
        if elevator.direction == ElevatorState.ASCENDING:
            return any(bp > elevator.cur_floor for bp in elevator.buttons_pressed)
        elif elevator.direction == ElevatorState.DESCENDING:
            return any(bp < elevator.cur_floor for bp in elevator.buttons_pressed)

    def get_action(self, state):
        new_calls = self._get_new_calls(state['up_calls'], state['down_calls'])
        for floor in new_calls:
            for idx, (zone_min, zone_max) in enumerate(self.zones):
                if floor < zone_max:
                    self.elevator_queues[idx].append(floor)
                    break

        # Preserve current state
        self.prev_up_calls = deepcopy(state['up_calls'])
        self.prev_down_calls = deepcopy(state['down_calls'])

        actions = []
        for idx, elevator in enumerate(state['elevators']):
            if elevator.cur_floor in self.elevator_queues[idx]:
                # Must let passenger board
                actions.append(ElevatorState.STOPPED)
                self.elevator_queues[idx].remove(elevator.cur_floor)
                self.prev_up_calls[elevator.cur_floor] = set()
                self.prev_down_calls[elevator.cur_floor] = set()
            elif elevator.cur_floor in elevator.buttons_pressed:
                # Passengers must be allowed to disembark
                actions.append(ElevatorState.STOPPED)
            elif self._button_press_in_direction(elevator):
                # Continue travelling in direction
                actions.append(elevator.direction)
            elif len(elevator.buttons_pressed) > 0:
                # Invert direction to serve boarded passengers before picking up new passngers (FCFS)
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.direction == ElevatorState.DESCENDING
                    else ElevatorState.DESCENDING
                )
            elif len(self.elevator_queues[idx]) > 0:
                # Pick up passengers waiting in hall, serve call
                pickup_floor = self.elevator_queues[idx][0]
                actions.append(
                    ElevatorState.ASCENDING
                    if elevator.cur_floor < pickup_floor
                    else ElevatorState.DESCENDING
                )
            else:
                # No boarded passengers, no passengers in queue
                # Return to center of zone
                zone_min, zone_max = self.zones[idx]
                zone_center = (zone_max - zone_min) // 2
                if elevator.cur_floor == zone_center:
                    actions.append(ElevatorState.STOPPED)
                else:
                    actions.append(
                        ElevatorState.ASCENDING
                        if elevator.cur_floor < zone_center
                        else ElevatorState.DESCENDING
                    )
        return actions

    def perform_update(self, state, action, reward, new_state):
        """Unused for benchmark agent that doesn't learn from experience."""
        pass