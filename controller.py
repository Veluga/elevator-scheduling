class Controller:
    def __init__(self, building, caller, agent, visualization=None, timesteps=10000):
        self.building = building
        self.caller = caller
        self.agent = agent
        self.visualization = visualization
        self.t = 0
        self.timesteps = timesteps

    def run(self):
        while self.t < self.timesteps:
            call_floor, destination_floor = self.caller.generate_call()
            if call_floor is not None and destination_floor is not None:
                self.building.call(call_floor, destination_floor)
            state = self.building.sample_state()
            action = self.agent.get_action(state)
            reward = self.building.perform_action(action)
            new_state = self.building.sample_state()
            if self.visualization:
                self.visualization.next_reward(reward)
            self.t += 1
        if self.visualization:
            self.visualization.display()

if __name__ == "__main__":
    from caller.continuous_random_call import ContinuousRandomCallCaller
    from building.parallel_elevator_moving import ParallelElevatorMoving
    from agent.tabular_q_learning import TabularQLearningAgent
    from visualization.average_reward import AverageReward

    building = ParallelElevatorMoving()
    caller = ContinuousRandomCallCaller()
    agent = TabularQLearningAgent()
    viz = AverageReward(sliding_window_size=10000)

    ctrl = Controller(building, caller, agent, visualization=None, timesteps=1000000)
    ctrl.run()