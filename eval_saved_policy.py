from caller.interfloor_caller import InterfloorCaller
from caller.up_peak_caller import UpPeakCaller
from caller.down_peak_caller import DownPeakCaller
from building.tf_building import TFBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from benchmark_controller import generate_available_actions
from copy import deepcopy

from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
import tensorflow as tf
import sys


available_actions = generate_available_actions()

# Building initialization
#caller = InterfloorCaller()
#caller = UpPeakCaller()
caller = DownPeakCaller()
eval_py_building = TFBuilding(DiscreteFloorTransition(caller), available_actions)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_building)

saved_policy = tf.compat.v2.saved_model.load(sys.argv[1])
time_step = eval_env.reset()
episode_return = 0.0

tf.compat.v1.enable_v2_behavior()

while not time_step.is_last():
    raw_env = eval_env.envs[0].building
    _, old_state = raw_env.sample_state()
    old_state = deepcopy(old_state)

    action_step = saved_policy.action(time_step)
    time_step = eval_env.step(action_step.action)
    episode_return += time_step.reward

    old_state['elevators'] = [(e.cur_floor, e.buttons_pressed, e.state) for e in old_state['elevators']]
    action = available_actions[action_step.action[0]]
    """ print("-----------------------------------------------------------------------")
    print("Previous Up Calls: {}".format(old_state['up_calls']))
    print("Previous Down Calls: {}".format(old_state['down_calls']))
    print("Previous Elevators: {}".format(old_state['elevators']))
    print("Action: {}".format(action)) """
print("Episode return: %i" % episode_return)