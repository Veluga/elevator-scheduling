from .agent import Agent
from abc import ABC, abstractmethod
from random import random, sample, shuffle
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

def linear(z):
    return z

class DifferentiableParameterizedFunction(ABC):
    def perform_update(self, step_size):
        pass


class ArtificialNeuralNetwork(DifferentiableParameterizedFunction):
    def __init__(self, num_layers, dims, activation_functions):
        self.num_layers = num_layers
        self.layers = {}
        self.outputs = {}
        self.activation_functions = activation_functions
        for i in range(self.num_layers):
            d0, d1 = dims[i]
            self.layers[i] = np.random.rand(d0, d1) * 0.001

    def build_input(self, state, action):
        bias = np.array([1], dtype=int)
        up_calls = np.array([state["up_calls"]], dtype=int)
        down_calls = np.array([state["down_calls"]], dtype=int)
        input_ = np.append(bias, up_calls)
        input_ = np.append(input_, down_calls)
        for cur_floor, elevator_state, buttons_pressed in state["elevators"]:
            # One hot encoding cur_floor
            input_ = np.append(input_, np.array([i == cur_floor for i in range(len(state["up_calls"]))], dtype=int))
            # One hot encoding elevator_state
            input_ = np.append(input_, np.array([i == elevator_state for i in range(1, 4)], dtype=int))
            input_ = np.append(input_, buttons_pressed)
        # One hot encoding actions
        input_ = np.append(input_, np.array([i == action for i in range(1, 4)], dtype=int))
        # TODO Hack for second dimension to be non-zero
        input_ = np.transpose(np.array([input_]))
        return input_

    def predict(self, state, action):
        input_ = self.build_input(state, action)
        for i in range(self.num_layers):
            self.outputs[i] = input_
            z = np.dot(self.layers[i], input_)
            input_ = self.activation_functions[i](z)
        return input_
    
    def gradient_step(self, step_size):
        delta = np.eye(1)
        for i in reversed(range(self.num_layers)):
            if i == self.num_layers-1:
                self.layers[i] += step_size * np.dot(delta, np.transpose(self.outputs[i]))
            else:
                delta = np.dot(np.transpose(self.layers[i+1]), delta)
                delta = np.dot(1-np.transpose(self.outputs[i+1]), delta)
                delta = np.dot(self.outputs[i+1], delta)
                self.layers[i] += step_size * np.dot(delta, np.transpose(self.outputs[i]))

    def perform_update(self, step_size):
        self.gradient_step(step_size)

class DifferentialSemiGradientSarsa(Agent):
    def __init__(self, q, available_actions, alpha=0.01, beta=0.01, epsilon=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.r_est = 0
        self.q = q
        self.available_actions = available_actions
        self.last_action = sample(self.available_actions, 1)[0]

    def get_action(self, state, get_last=True):
        if get_last:
            return [self.last_action]

        if random() < self.epsilon:
            self.last_action = sample(self.available_actions, 1)[0]
        else:
            max_q = float('-inf')
            max_q_action = None
            shuffle(self.available_actions)
            for action in self.available_actions:
                q_est = self.q.predict(state, action)
                if q_est > max_q:
                    max_q = q_est
                    max_q_action = action
            self.last_action = max_q_action
        return [self.last_action]

    def perform_update(self, state, action, reward, next_state):
        q_state = self.q.predict(state, action[0])
        next_action = self.get_action(next_state, get_last=False)
        q_next_state = self.q.predict(next_state, next_action[0])
        
        delta = sum(reward) - self.r_est + q_next_state - q_state
        self.r_est += self.beta * delta
        self.q.perform_update(self.alpha * delta)