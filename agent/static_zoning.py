from .benchmark_agent import BenchmarkAgent
from building.building import ElevatorState
import settings as s

from collections import deque
from copy import deepcopy

class StaticZoningAgent(BenchmarkAgent):
    """Agent that implements a zoning strategy whereby every car serves only a set of neighbouring floors.
    Zones are determined statically upon initialization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zones = self._calculate_zones()

    def _calculate_zones(self, num_floors=s.NUM_FLOORS, num_elevators=s.NUM_ELEVATORS):
        """Divide building into `num_elevators` zones and assign them to the elevators."""
        serving_floors = []
        from_ = 0
        for i in range(num_elevators):
            floors_to_serve = int(num_floors / (num_elevators-i))
            serving_floors.append((from_, from_ + floors_to_serve))
            from_ += floors_to_serve
            num_floors -= floors_to_serve
        return serving_floors

    def assign_calls(self, new_up_calls, new_down_calls):
        """Calls will be assigned to elevator that serves zone which contains the source floor."""
        for floor in new_up_calls:
            for idx, (zone_min, zone_max) in enumerate(self.zones):
                if floor < zone_max:
                    self.elevator_up_queues[idx].append(floor)
                    break
        for floor in new_down_calls:
            for idx, (zone_min, zone_max) in enumerate(self.zones):
                if floor < zone_max:
                    self.elevator_down_queues[idx].append(floor)
                    break

    def handle_unused(self, idx, elevator):
        # Return to center of zone
        zone_min, zone_max = self.zones[idx]
        zone_center = (zone_max - zone_min) // 2
        if elevator.cur_floor == zone_center:
            return ElevatorState.STOPPED
        else:
            return ElevatorState.ASCENDING if elevator.cur_floor < zone_center else ElevatorState.DESCENDING