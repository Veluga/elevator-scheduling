from random import randint
from .agent import Agent

class RandomPolicyAgent(Agent):
    def __init__(self, available_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_actions = available_actions

    def get_action(self, _):
        return self.available_actions[randint(0, len(self.available_actions)-1)]

    def perform_update(self, state, action, reward, new_state):
        pass