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
            if t > 0.99 * self.timesteps:
                print(state, action)
        if self.visualization:
            self.visualization.display()

if __name__ == "__main__":
    from caller.continuous_random_call import ContinuousRandomCallCaller
    from building.discrete_floor_transition import DiscreteFloorTransition
    from agent.tabular_q_learning import TabularQLearningAgent
    
    from visualization.average_reward import AverageReward
    from visualization.cumulative_reward import CumulativeReward

    building = DiscreteFloorTransition()
    caller = ContinuousRandomCallCaller()
    agent = TabularQLearningAgent()
    viz = AverageReward(sliding_window_size=10000)
    #viz = CumulativeReward()
    #viz = None

    ctrl = Controller(building, caller, agent, visualization=viz, timesteps=1000000)
    ctrl.run()