import settings as s
from .building import Building, Call, Elevator, ElevatorState

class ParallelElevatorMoving(Building):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up_calls = {floor_num: set() for floor_num in range(self.floors)}
        self.down_calls = {floor_num: set() for floor_num in range(self.floors)}

    def call(self, call_floor, destination_floor):
        if call_floor < destination_floor:
            self.up_calls[call_floor].add((destination_floor, 0))
        else:
            self.down_calls[call_floor].add((destination_floor, 0))

    def sample_state(self):
        # Up/Down calls
        # Positions and directions
        # Velocities
        up_calls = tuple([len(s) > 0 for f, s in self.up_calls.items()])
        down_calls = tuple([len(s) > 0 for f, s in self.down_calls.items()])
        positions = tuple([e.cur_floor for e in self.elevators])
        velocities = tuple([e.cur_velocity for e in self.elevators])
        return up_calls + down_calls + positions + velocities

    def perform_action(self, actions):
        rewards = []
        for elevator, action in zip(self.elevators, actions):
            if action == ElevatorState.ASCENDING:
                if elevator.cur_floor == self.floors-1:
                    # Cannot ascend from top floor
                    rewards.append(-10)
                    continue
                elevator.stopped = False
                elevator.direction = ElevatorState.ASCENDING
                elevator.interfloor_distance += elevator.cur_velocity * s.TICK_LENGTH_IN_SECONDS
                elevator.cur_floor += elevator.interfloor_distance // self.floor_height
                elevator.interfloor_distance %= self.floor_height
                elevator.cur_velocity = min(
                    elevator.max_velocity,
                    elevator.cur_velocity + elevator.acceleration * s.TICK_LENGTH_IN_SECONDS
                )
            elif action == ElevatorState.DESCENDING:
                if elevator.cur_floor == 0:
                    # Cannot descend from ground floor
                    rewards.append(-10)
                    continue
                elevator.stopped = False
                elevator.direction = ElevatorState.DESCENDING
                elevator.interfloor_distance += elevator.cur_velocity * s.TICK_LENGTH_IN_SECONDS
                elevator.cur_floor -= elevator.interfloor_distance // self.floor_height
                elevator.interfloor_distance %= self.floor_height
                elevator.cur_velocity = min(
                    elevator.max_velocity,
                    elevator.cur_velocity + elevator.acceleration * s.TICK_LENGTH_IN_SECONDS
                )
            else:
                # Elevator intends to stop, decelerate
                elevator.interfloor_distance += elevator.cur_velocity * s.TICK_LENGTH_IN_SECONDS
                if elevator.direction == ElevatorState.ASCENDING:
                    elevator.cur_floor += elevator.interfloor_distance // self.floor_height
                else:
                    elevator.cur_floor -= elevator.interfloor_distance // self.floor_height
                elevator.interfloor_distance %= self.floor_height
                elevator.cur_velocity = max(
                    0,
                    elevator.cur_velocity - elevator.acceleration * s.TICK_LENGTH_IN_SECONDS
                )

                if elevator.cur_velocity <= 0.25 and elevator.interfloor_distance <= 0.25:
                    # Arbitrary criteria to accept stopping
                    elevator.stopped = True
                    for destination, waiting_time in self.up_calls[elevator.cur_floor]:
                        elevator.buttons_pressed.add(destination)
                    for destination, waiting_time in self.down_calls[elevator.cur_floor]:
                        elevator.buttons_pressed.add(destination)
                    if elevator.cur_floor in elevator.buttons_pressed:
                        elevator.buttons_pressed.remove(cur_floor)
                        # TODO Make this dependent on waiting times
                        rewards.append(1)
                        continue
            rewards.append(0)
        return rewards