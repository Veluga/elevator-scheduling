from caller.caller import Caller
from environment.environment import Environment
from agent.agent import Agent

class Controller:
    def __init__(self, environment, caller, agent):
        self.environment = environment
        self.caller = caller
        self.agent = agent

    def run(self):
        pass

if __name__ == "__main__":
    pass