from math import sqrt
from copy import deepcopy
from building.discrete_floor_transition import ElevatorState
import settings as s

import numpy as np
import random


def generate_available_actions(num_elevators=s.NUM_ELEVATORS):
    """Generate list of available actions (which are lists of ElevatorState values)."""
    def helper(available_actions, accu, remaining_elevators):
        if remaining_elevators == 0:
            available_actions.append(accu)
            return available_actions
        
        res = []
        for action in ElevatorState:
            helper(available_actions, deepcopy(accu) + [action], remaining_elevators-1)
        return available_actions
    return helper([], [], num_elevators)

class Controller:
    """The Controller ties together the building (environment) and agent.
    Agents receive state samples to generate actions, which are passed to the building for state updates.
    """
    def __init__(self, building, agent, visualization=None, timesteps=10000):
        self.building = building
        self.agent = agent
        self.visualization = visualization
        self.timesteps = timesteps

    def run(self):
        for t in range(self.timesteps):
            # Standard RL loop of sample-act-sample-update
            _, state = self.building.sample_state()
            action = self.agent.get_action(state)
            reward = self.building.perform_action(action)
            _, new_state = self.building.sample_state()
            agent.perform_update(state, action, reward, new_state)
            if self.visualization:
                # Pass reward to visualization
                self.visualization.next_reward(reward)
            if t % int(sqrt(self.timesteps)) == 0:
                # Progress "bar"
                print("Time {}/{}".format(t, self.timesteps))
        if self.visualization:
            self.visualization.display()

if __name__ == "__main__":
    random.seed(s.RANDOM_SEED)
    np.random.seed(s.RANDOM_SEED)

    from caller.get_caller import get_caller

    from building.discrete_floor_transition import DiscreteFloorTransition, ElevatorState

    from agent.benchmark_agent import get_benchmark_agent

    from visualization.average_reward import AverageReward
    from visualization.cumulative_reward import CumulativeReward


    caller = get_caller()
    building = DiscreteFloorTransition(caller, track_passengers=True)
    available_actions = generate_available_actions()
    agent = get_benchmark_agent(available_actions)
    viz = None

    ctrl = Controller(building, agent, visualization=viz, timesteps=s.EPISODE_LENGTH)
    ctrl.run()
    print("Delivered passengers: {}%".format(
        len([p for p in ctrl.building.passengers.values() if p.served]) / len(ctrl.building.passengers.values()) * 100
    ))
    print("Average waiting time: {}".format(
        sum([p.waiting_time for p in ctrl.building.passengers.values()]) / len([p for p in ctrl.building.passengers.values() if p.served])
    ))
    print("Squared average waiting time: {}".format(
        sum([p.waiting_time**2 for p in ctrl.building.passengers.values()]) / len([p for p in ctrl.building.passengers.values() if p.served])
    ))
    print("Average system time: {}".format(
        sum([p.system_time for p in ctrl.building.passengers.values()]) / len([p for p in ctrl.building.passengers.values() if p.served])
    ))
    print("Squared average system time: {}".format(
        sum([p.system_time**2 for p in ctrl.building.passengers.values()]) / len([p for p in ctrl.building.passengers.values() if p.served])
    ))
    print(">60 seconds waiting time: {}%".format(
        len([p for p in ctrl.building.passengers.values() if p.waiting_time > 60]) / len(ctrl.building.passengers) * 100
    ))
    print("Total reward collected: {}".format(ctrl.building._total_reward))