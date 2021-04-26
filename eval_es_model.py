import settings as s
from caller.get_caller import get_caller
from building.tf_building import TFBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from benchmark_controller import generate_available_actions
from es_controller import ANN

from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from copy import deepcopy
import tensorflow as tf
import numpy as np
import sys


def eval_model(path, policy_num, num_episodes):
    """Given path to stored model (./experiments/...) and the saved policy step, evaluate model performance over a number of episodes."""

    available_actions = generate_available_actions()

    # Building initialization
    caller = get_caller()
    building = DiscreteFloorTransition(caller, track_passengers=True)

    # Restore policy
    saved_policy = np.load(path + f"/weights/policy_{policy_num}.npy", allow_pickle=True)
    saved_policy = np.asscalar(saved_policy)

    waiting_times = []
    squared_waiting_times = []
    system_times = []
    squared_system_times = []
    unacceptable_waiting_times = []

    for i in range(num_episodes):
        #print(f"Episode {i}/{num_episodes}...")
        building.reset()
        episode_return = 0.0
        timestep = 0

        while timestep < s.EPISODE_LENGTH:
            # Sample-act-reward
            state_vector, _ = building.sample_state()
            action_idx = saved_policy.predict(state_vector)
            reward = building.perform_action(available_actions[action_idx])
            episode_return += sum(reward)
            timestep += 1

        waiting_times.append(
            sum([p.waiting_time for p in building.passengers.values()]) / len([p for p in building.passengers.values()])
        )
        squared_waiting_times.append(
            sum([p.waiting_time**2 for p in building.passengers.values()]) / len([p for p in building.passengers.values()])
        )
        system_times.append(
            sum([p.system_time for p in building.passengers.values()]) / len([p for p in building.passengers.values()])    
        )
        squared_system_times.append(
            sum([p.system_time**2 for p in building.passengers.values()]) / len([p for p in building.passengers.values()])    
        )
        unacceptable_waiting_times.append(
            len([p for p in building.passengers.values() if p.waiting_time > 60]) / len(building.passengers) * 100
        )

    print(f"----- {policy_num} -----")
    print("Average waiting time: {}".format(
        sum(waiting_times) / len(waiting_times)
    ))
    print("Squared average waiting time: {}".format(
        sum(squared_waiting_times) / len(squared_waiting_times)
    ))
    print("Average system time: {}".format(
        sum(system_times) / len(system_times)
    ))
    print("Squared average system time: {}".format(
        sum(squared_system_times) / len(squared_system_times)
    ))
    print(">60 seconds waiting time: {}%".format(
        sum(unacceptable_waiting_times) / len(unacceptable_waiting_times)
    ))
    return {
        'waiting_times': waiting_times,
        'squared_waiting_times': squared_waiting_times,
        'system_times': system_times,
        'squared_system_times': squared_system_times,
        'unacceptable_waiting_times': unacceptable_waiting_times
    }

if __name__ == "__main__":
    experiment_path = sys.argv[1]
    policy_step = int(sys.argv[2])
    num_episodes = int(sys.argv[3])

    results = eval_model(experiment_path, policy_step, num_episodes)