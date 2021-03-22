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
    #random.seed(s.RANDOM_SEED)
    #np.random.seed(s.RANDOM_SEED)

    from caller.interfloor_caller import InterfloorCaller
    from caller.up_peak_caller import UpPeakCaller
    from caller.down_peak_caller import DownPeakCaller
    from building.discrete_floor_transition import DiscreteFloorTransition, ElevatorState
    from agent.random_policy import RandomPolicyAgent
    from agent.round_robin import RoundRobinAgent
    from agent.static_zoning import StaticZoningAgent
    from agent.up_peak_ges import UpPeakGES

    from visualization.average_reward import AverageReward
    from visualization.cumulative_reward import CumulativeReward

    #caller = InterfloorCaller()
    caller = UpPeakCaller()
    #caller = DownPeakCaller()
    
    building = DiscreteFloorTransition(caller)
    
    available_actions = generate_available_actions()
    
    #agent = RandomPolicyAgent(available_actions)
    agent = RoundRobinAgent()
    #agent = StaticZoningAgent()
    #agent = UpPeakGES()
    
    #viz = AverageReward(sliding_window_size=100)
    viz = CumulativeReward()
    #viz = None

    ctrl = Controller(building, agent, visualization=viz, timesteps=3600)
    ctrl.run()
    print("Delivered Passengers: {}%".format(ctrl.visualization.cumulative_reward / 2 / building._generated_calls))