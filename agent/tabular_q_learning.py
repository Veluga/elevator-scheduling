import settings as s
from .agent import Agent
from building.building import ElevatorState
from random import random, sample

class TabularQLearningAgent(Agent):
    def __init__(self, gamma=s.DISCOUNT_RATE, alpha=s.STEP_SIZE, epsilon=s.EPSILON, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q = {}
        self.last_state_action_pair = None

    def get_estimated_best_action(self, state):
        max_val = float('-inf')
        max_action = None
        for action in self.get_available_actions(self.elevators, ()):
            if self.q[(state, action)] > max_val:
                max_val = self.q[(state, action)]
                max_action = action
        return max_action, max_val

    def get_available_actions(self, elevators, accu):
        if elevators == 0:
            yield accu
            return
        for action in ElevatorState:
            yield from self.get_available_actions(elevators-1, accu + (action,))

    def init_action_values(self, state):
        for action in self.get_available_actions(self.elevators, ()):
            self.q[(state, action)] = 0
        # Hack to make checking for stored action values easier
        self.q[state] = True

    def get_action(self, state):
        state = state['up_calls'] + state['down_calls'] + state['elevators']
        if state not in self.q:
            self.init_action_values(state)
        
        if random() < self.epsilon:
            exploratory_action = sample(
                list(self.get_available_actions(self.elevators, ())), 1
            )[0]
            return exploratory_action
        else:
            max_action, _ =  self.get_estimated_best_action(state)
            return max_action

    def perform_update(self, state, action, reward, new_state):
        state = state['up_calls'] + state['down_calls'] + state['elevators']
        new_state = new_state['up_calls'] + new_state['down_calls'] + new_state['elevators']
        if new_state not in self.q:
            self.init_action_values(new_state)
        _, max_val = self.get_estimated_best_action(new_state)
        self.q[(state, action)] += self.alpha * (sum(reward) + self.gamma * max_val - self.q[(state, action)])