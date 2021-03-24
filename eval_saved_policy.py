from caller.interfloor_caller import InterfloorCaller
from caller.up_peak_caller import UpPeakCaller
from caller.down_peak_caller import DownPeakCaller
from caller.mixed_caller import MixedCaller
from building.tf_building import TFBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from benchmark_controller import generate_available_actions
from copy import deepcopy

from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
import tensorflow as tf
import numpy as np
import sys


"""Given path to stored model (./experiments/.../weights/policy_XYZ), evaluate model performance over one episode."""

available_actions = generate_available_actions()

# Building initialization
#caller = DownPeakCaller()
#caller = InterfloorCaller()
#caller = UpPeakCaller()
caller = MixedCaller()

eval_py_building = TFBuilding(DiscreteFloorTransition(caller), available_actions)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_building)

# Restore policy
saved_policy = tf.compat.v2.saved_model.load(sys.argv[1])
time_step = eval_env.reset()
episode_return = 0.0

tf.compat.v1.enable_v2_behavior()

while not time_step.is_last():
    # Sample-act-sample-reward
    raw_env = eval_env.envs[0].building
    _, old_state = raw_env.sample_state()
    old_state = deepcopy(old_state)

    action_step = saved_policy.action(time_step)
    time_step = eval_env.step(action_step.action)
    episode_return += time_step.reward

    old_state['elevators'] = [(e.cur_floor, e.buttons_pressed, e.state) for e in old_state['elevators']]
    action = available_actions[action_step.action[0]]

building = eval_env.envs[0].building
print("Delivered passengers: {}%".format(
    len([p for p in building.passengers.values() if p.served]) / len(building.passengers.values()) * 100
))
print("Average waiting time: {}".format(
    sum([p.waiting_time for p in building.passengers.values()]) / len([p for p in building.passengers.values() if p.served])
))
print("Squared average waiting time: {}".format(
    sum([p.waiting_time**2 for p in building.passengers.values()]) / len([p for p in building.passengers.values() if p.served])
))
print("Average system time: {}".format(
    sum([p.system_time for p in building.passengers.values()]) / len([p for p in building.passengers.values() if p.served])
))
print("Squared average system time: {}".format(
    sum([p.system_time**2 for p in building.passengers.values()]) / len([p for p in building.passengers.values() if p.served])
))
print(">60 seconds waiting time: {}%".format(
    len([p for p in building.passengers.values() if p.waiting_time > 60]) / len(building.passengers) * 100
))