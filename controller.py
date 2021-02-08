from caller.caller import Caller
from environment.environment import Environment
from agent.agent import Agent

class Controller:
    def __init__(self, environment, caller, agent):
        self.environment = environment
        self.caller = caller
        self.agent = agent

    def run(self):
        while True:
            floor, direction = self.caller.generate_call()
            if floor is not None and direction is not None:
                self.environment.call(floor, direction)
            state = self.environment.sample_state()
            action = self.agent.get_action(state)
            reward = self.environment.perform_action(action)
            new_state = self.environment.sample_state()
            self.agent.perform_update(state, action, reward, new_state)

if __name__ == "__main__":
    building = Environment()
    caller = Caller()
    agent = Agent()

    ctrl = Controller(building, caller, agent)
    ctrl.run()