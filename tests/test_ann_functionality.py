from ..agent.differential_semi_gradient_sarsa import ArtificialNeuralNetwork, linear, sigmoid
from ..building.discrete_floor_transition import DiscreteFloorTransition
from ..building.building import ElevatorState
from random import randint
from copy import deepcopy
import numpy as np
    

def test_ann_constructor():
    d0, d1 = randint(1, 100), randint(1, 100)
    ann = ArtificialNeuralNetwork(1, [(d0, d1)], {0: linear})
    assert ann.num_layers == 1
    assert len(ann.layers) == 1
    assert ann.layers[0].shape == (d0, d1)
    assert ann.activation_functions[0] == linear

def test_sigmoid():
    m = np.array([-1000, 0, 1000])
    assert np.all(sigmoid(m) == np.array([0, 0.5, 1]))

def test_linear():
    m = np.random.rand(10, 10)
    assert np.all(linear(m) == m)

def test_build_input():
    building = DiscreteFloorTransition(floors=5, elevators=1)
    ann = ArtificialNeuralNetwork(1, [(10, 10)], {0: linear})
    
    state = building.sample_state()
    action = ElevatorState.STOPPED
    input_ = ann.build_input(state, action)

    # Bias
    assert input_[0] == 1
    # Up Calls
    assert np.all(input_[1:6] == 0)
    # Down Calls
    assert np.all(input_[6:11] == 0)
    # Current Floor
    assert input_[11] == 1
    assert np.all(input_[12:16] == 0)
    # Elevator State
    assert input_[16] == 1
    assert np.all(input_[17:19] == 0)
    # Buttons Pressed
    assert np.all(input_[19:24] == 0)
    # Action
    assert input_[24] == 1
    assert np.all(input_[25:27] == 0)

def test_predict():
    building = DiscreteFloorTransition(floors=5, elevators=1)
    ann = ArtificialNeuralNetwork(1, [(1, 27)], {0: linear})
    ann.layers[0] = np.zeros((1, 27))
    
    state = building.sample_state()
    action = ElevatorState.STOPPED
    predicted_q_value = ann.predict(state, action)
    
    assert predicted_q_value == 0
    assert len(ann.outputs) == 1
    assert ann.outputs[0].shape == (27, 1)
    assert predicted_q_value.shape == (1, 1)

def test_gradient_step():
    building = DiscreteFloorTransition(floors=5, elevators=1)
    ann = ArtificialNeuralNetwork(3, [(50, 27), (100, 50), (1, 100)], {0: sigmoid, 1: sigmoid, 2: linear})
    
    state = building.sample_state()
    action = ElevatorState.STOPPED

    layers_before_update = deepcopy(ann.layers)
    ann.predict(state, action)
    ann.gradient_step(0.1)
    layers_after_update = deepcopy(ann.layers)

    for layer_num in ann.layers.keys():
        assert np.any(
            layers_before_update[layer_num] != layers_after_update[layer_num]
        )