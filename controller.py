from math import sqrt

class Controller:
    def __init__(self, building, caller, agent, visualization=None, timesteps=10000):
        self.building = building
        self.caller = caller
        self.agent = agent
        self.visualization = visualization
        self.timesteps = timesteps

    def run(self):
        for t in range(self.timesteps):
            call_floor, destination_floor = self.caller.generate_call()
            if call_floor is not None and destination_floor is not None:
                self.building.call(call_floor, destination_floor)
            state = self.building.sample_state()
            action = self.agent.get_action(state)
            reward = self.building.perform_action(action)
            new_state = self.building.sample_state()
            agent.perform_update(state, action, reward, new_state)
            if self.visualization:
                self.visualization.next_reward(reward)
            if t % int(sqrt(self.timesteps)) == 0:
                print("Time {}/{}".format(t, self.timesteps))
        if self.visualization:
            self.visualization.display()

if __name__ == "__main__":
    from caller.continuous_random_call import ContinuousRandomCallCaller
    from building.discrete_floor_transition import DiscreteFloorTransition, ElevatorState
    from agent.tabular_q_learning import TabularQLearningAgent
    from agent.differential_semi_gradient_sarsa import DifferentialSemiGradientSarsa, ArtificialNeuralNetwork, sigmoid, linear

    from visualization.average_reward import AverageReward
    from visualization.cumulative_reward import CumulativeReward

    building = DiscreteFloorTransition()
    caller = ContinuousRandomCallCaller()
    #agent = TabularQLearningAgent()
    ann = ArtificialNeuralNetwork(1, [(1, 27)], {0: linear})
    agent = DifferentialSemiGradientSarsa(q=ann, available_actions=list(ElevatorState))
    #viz = AverageReward(sliding_window_size=100)
    viz = CumulativeReward()
    #viz = None

    ctrl = Controller(building, caller, agent, visualization=viz, timesteps=10000)
    ctrl.run()