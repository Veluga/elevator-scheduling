from ..building.building import ElevatorState
from ..building.discrete_floor_transition import DiscreteFloorTransition
from ..caller.interfloor_caller import InterfloorCaller
from ..settings import NUM_FLOORS, NUM_ELEVATORS, REWARD_DELIVERED_PASSENGER

from copy import deepcopy
import pytest
import numpy as np


@pytest.fixture
def building():
    caller = InterfloorCaller()
    building = DiscreteFloorTransition(caller=caller, floors=NUM_FLOORS, elevators=NUM_ELEVATORS)
    yield building

def test_reset(building):
    building.reset()

    assert all(building.up_calls[f] == set() for f in range(NUM_FLOORS))
    assert all(building.down_calls[f] == set() for f in range(NUM_FLOORS))
    assert all(elevator.state == ElevatorState.STOPPED for elevator in building.elevators)
    assert all(elevator.cur_floor == 0 for elevator in building.elevators)
    assert all(elevator.buttons_pressed == set() for elevator in building.elevators)

def test_call(building):
    building.call(0, 1)
    building.call(4, 3)

    assert len(building.up_calls[0]) == 1
    assert len(building.down_calls[4]) == 1

def test_sample_state(building):
    building.reset()
    building.call(0, 1)
    building.call(1, 0)

    state_vector, state_dict = building.sample_state()
    
    # up_calls, down_calls, (cur_floor, state, buttons_pressed) * NUM_ELEVATORS
    assert state_vector.shape == (NUM_FLOORS + NUM_FLOORS + NUM_ELEVATORS * (1 + 3 + NUM_FLOORS),)
    assert np.all(state_vector[0:NUM_FLOORS] == [1] + [0] * (NUM_FLOORS-1))
    assert np.all(state_vector[NUM_FLOORS:NUM_FLOORS*2] == [0, 1] + [0] * (NUM_FLOORS-2))
    
    assert len(state_dict) == 3
    assert len(state_dict['up_calls']) == len(building.up_calls)
    assert len(state_dict['down_calls']) == len(building.down_calls)
    assert len(state_dict['elevators']) == len(building.elevators)

    for elevator in building.elevators:
        elevator.cur_floor = NUM_FLOORS-1
        elevator.state = ElevatorState.ASCENDING
        elevator.buttons_pressed.add(0)
    
    state_vector, _ = building.sample_state()
    
    assert np.all(
        state_vector[NUM_FLOORS*2:NUM_FLOORS*2 + 4 + NUM_FLOORS] == [NUM_FLOORS-1, 0, 1, 0] + [1] + [0] * (NUM_FLOORS-1)
    )

def test_descending_from_ground_floor_does_not_change_elevator_state(building):
    prev_elevators = deepcopy(building.elevators)
    building.perform_action([ElevatorState.DESCENDING] * NUM_ELEVATORS)
    new_elevators = deepcopy(building.elevators)
    
    assert (prev_elevators[i] == new_elevators[i] for i in range(len(prev_elevators)))

def test_ascending_from_top_floor_does_not_change_elevator_state(building):
    for elevator in building.elevators:
        elevator.cur_floor = NUM_FLOORS-1
    
    prev_elevators = deepcopy(building.elevators)
    building.perform_action([ElevatorState.ASCENDING] * NUM_ELEVATORS)
    new_elevators = deepcopy(building.elevators)
    
    assert (prev_elevators[i] == new_elevators[i] for i in range(len(prev_elevators)))

def test_picking_up_passenger(building):
    building.call(0, 1)
    building.perform_action([ElevatorState.STOPPED] * NUM_ELEVATORS)

    assert all(building.up_calls[f] == set() for f in range(NUM_FLOORS))
    assert any(elevator.buttons_pressed == set([1]) for elevator in building.elevators)

def test_reward_format(building):
    rewards = building.perform_action([ElevatorState.STOPPED] * NUM_ELEVATORS)

    assert len(rewards) == NUM_ELEVATORS

def test_position_update(building):
    building.perform_action([ElevatorState.ASCENDING] * NUM_ELEVATORS)
    assert all(elevator.cur_floor == 1 for elevator in building.elevators)

    building.perform_action([ElevatorState.STOPPED] * NUM_ELEVATORS)
    assert all(elevator.cur_floor == 1 for elevator in building.elevators)

    building.perform_action([ElevatorState.DESCENDING] * NUM_ELEVATORS)
    assert all(elevator.cur_floor == 0 for elevator in building.elevators)

def test_reward_for_delivering_passenger(building):
    building.call(0, 1)
    building.perform_action([ElevatorState.STOPPED] * NUM_ELEVATORS)
    building.perform_action([ElevatorState.ASCENDING] * NUM_ELEVATORS)
    rewards = building.perform_action([ElevatorState.STOPPED] * NUM_ELEVATORS)

    assert sum(rewards) == REWARD_DELIVERED_PASSENGER