from math import sqrt
from copy import deepcopy
import settings as s


def generate_available_actions(num_elevators=s.NUM_ELEVATORS):
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
    def __init__(self, building, caller, agent, visualization=None, timesteps=10000):
        self.building = building
        self.caller = caller
        self.agent = agent
        self.visualization = visualization
        self.timesteps = timesteps

    def run(self):
        for t in range(self.timesteps):
            _, state = self.building.sample_state()
            action = self.agent.get_action(state)
            reward = self.building.perform_action(action)
            _, new_state = self.building.sample_state()
            agent.perform_update(state, action, reward, new_state)
            if self.visualization:
                self.visualization.next_reward(reward)
            if t % int(sqrt(self.timesteps)) == 0:
                print("Time {}/{}".format(t, self.timesteps))
        if self.visualization:
            self.visualization.display()

if __name__ == "__main__":
    from caller.interfloor_caller import InterfloorCaller
    from caller.up_peak_caller import UpPeakCaller
    from caller.down_peak_caller import DownPeakCaller
    from building.discrete_floor_transition import DiscreteFloorTransition, ElevatorState
    from agent.random_policy import RandomPolicyAgent
    from agent.round_robin import RoundRobinAgent

    from visualization.average_reward import AverageReward
    from visualization.cumulative_reward import CumulativeReward

    caller = InterfloorCaller()
    #caller = UpPeakCaller()
    #caller = DownPeakCaller()
    
    building = DiscreteFloorTransition(caller)
    
    available_actions = generate_available_actions()
    
    #agent = RandomPolicyAgent(available_actions)
    agent = RoundRobinAgent()
    
    #viz = AverageReward(sliding_window_size=100)
    viz = CumulativeReward()
    #viz = None

    ctrl = Controller(building, caller, agent, visualization=viz, timesteps=3600)
    ctrl.run()
    print("Delivered Passengers: {}%".format(ctrl.visualization.cumulative_reward / 2 / building._generated_calls))