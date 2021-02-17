import settings as s
from .building import Building, Call, Elevator, ElevatorState
from random import randint


class DiscreteFloorTransition(Building):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def call(self, call_floor, destination_floor):
        if call_floor < destination_floor:
            self.up_calls[call_floor].add(destination_floor)
        else:
            self.down_calls[call_floor].add(destination_floor)

    def sample_state(self):
        up_calls = tuple([len(s) > 0 for f, s in self.up_calls.items()])
        down_calls = tuple([len(s) > 0 for f, s in self.down_calls.items()])
        elevators = tuple(
            tuple([e.cur_floor, e.state, tuple([f in e.buttons_pressed for f in range(self.floors)])]) 
            for e in self.elevators
        )
        return {"up_calls": up_calls, "down_calls": down_calls, "elevators": elevators}

    def perform_action(self, actions):
        from_, to = self.caller.generate_call()
        if from_ is not None and to is not None:
            self.call(from_, to)
        
        actions = [actions]
        def update_position(elevator, action):
            if action == ElevatorState.ASCENDING:
                if elevator.cur_floor != self.floors-1:
                    elevator.cur_floor += 1
                else:
                    return -1
            elif action == ElevatorState.DESCENDING:
                if elevator.cur_floor != 0:
                    elevator.cur_floor -= 1
                else:
                    return -1
            return 0

        rewards = []
        for elevator, action in zip(self.elevators, actions):
            rewards.append(0)
            rewards[-1] += update_position(elevator, action)
            elevator.state = action
            
            # Only stopped elevators can have passengers boarding or disembarking
            if elevator.state != ElevatorState.STOPPED:
                continue
            
            # Check whether ascending passengers boarding
            if len(self.up_calls[elevator.cur_floor]) > 0:
                for destination in self.up_calls[elevator.cur_floor]:
                    elevator.buttons_pressed.add(destination)
                self.up_calls[elevator.cur_floor] = set()

            # Check whether descending passengers boarding
            if len(self.down_calls[elevator.cur_floor]) > 0:
                for destination in self.down_calls[elevator.cur_floor]:
                    elevator.buttons_pressed.add(destination)
                self.down_calls[elevator.cur_floor] = set()

            # Check whether passengers disembarking
            if elevator.cur_floor in elevator.buttons_pressed:
                rewards[-1] += 2
                elevator.buttons_pressed.remove(elevator.cur_floor)

        return rewards
    
    def _reset(self):
        def random_call_destinations(floor_num):
            if floor_num == self.floors-1:
                destination_floor = randint(0, floor_num-1)
                return set([destination_floor])
            elif floor_num == 0:
                destination_floor = randint(1, self.floors-1)
                return set([destination_floor])
            else:
                destination_floor = randint(0, self.floors-1)
                while destination_floor == floor_num:
                    destination_floor = randint(0, self.floors-1)
                return set([destination_floor])

        #self.up_calls = {floor_num: random_call_destinations(floor_num) for floor_num in range(self.floors)}
        #self.down_calls = {floor_num: random_call_destinations(floor_num) for floor_num in range(self.floors)}
        self.up_calls = {floor_num: set() for floor_num in range(self.floors)}
        self.down_calls = {floor_num: set() for floor_num in range(self.floors)}