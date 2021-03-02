import numpy as np
import settings as s
import random
import pathlib
import sys
from enum import Enum

class ANN:
    def __init__(self, input_dims, hidden_layer_units, num_actions, scale=s.NOISE_STANDARD_DEVIATION):
        # Input is extended by bias
        self.input_dims = input_dims
        self.hidden_layer_units = hidden_layer_units
        self.num_actions = num_actions

        self.input_layer = np.random.normal(size=(input_dims+1, hidden_layer_units[0]), scale=scale)
        self.hidden_layers = []
        for i in range(1, len(hidden_layer_units)):
            self.hidden_layers.append(
                np.random.normal(size=(hidden_layer_units[i-1], hidden_layer_units[i]), scale=scale)
            )
        self.output_layer = np.random.normal(size=(hidden_layer_units[-1], num_actions), scale=scale)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _relu(self, x):
        return np.maximum(x, 0)

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def predict(self, input_):
        # Bias extension
        input_ = np.append(input_, [1])
        input_ = np.transpose(input_)

        x = np.dot(input_, self.input_layer)
        x = self._relu(x)

        for layer in self.hidden_layers:
            x = np.dot(x, layer)
            x = self._relu(x)
        
        x = np.dot(x, self.output_layer)
        outputs = self._softmax(x)

        return np.argmax(outputs)

    def __add__(self, other):
        res = ANN(self.input_dims, self.hidden_layer_units, self.num_actions)
        res.input_layer = self.input_layer + other.input_layer
        
        for idx, (l, other_l) in enumerate(zip(self.hidden_layers, other.hidden_layers)):
            res.hidden_layers[idx] = l + other_l

        res.output_layer = self.output_layer + other.output_layer

        return res

    def __mul__(self, scalar):
        res = ANN(self.input_dims, self.hidden_layer_units, self.num_actions)
        res.input_layer = self.input_layer * scalar
        
        for idx, hl in enumerate(self.hidden_layers):
            res.hidden_layers[idx] = hl * scalar
        
        res.output_layer = self.output_layer * scalar
        
        return res

def evaluate_individual(individual, building, available_actions):
    building.reset()
    episode_reward = 0
    for _ in range(s.EPISODE_LENGTH):
        state_vec, _ = building.sample_state()
        action = available_actions[individual.predict(state_vec)]
        episode_reward += sum(building.perform_action(action))
    return episode_reward

""" class DummyEnvironment:
    class Action(Enum):
        LEFT = 0
        RIGHT = 1

    def __init__(self, num_states=1000):
        self.num_states = num_states
        self.current_state = 0

    def get_available_actions(self):
        return [DummyEnvironment.Action.LEFT, DummyEnvironment.Action.RIGHT]
    
    def perform_action(self, action):
        reward = 0
        if self.current_state == self.num_states-1:
            reward = 1
        if action == DummyEnvironment.Action.LEFT:
            self.current_state -= 1
        else:
            self.current_state += 1
        self.current_state %= self.num_states
        return reward

    def sample_state(self):
        return np.array([self.current_state])
    
    def reset(self):
        self.current_state = 0

def evaluate_individual(individual, env, available_actions):
    env.reset()
    episode_reward = 0
    for _ in range(s.EPISODE_LENGTH):
        state = env.sample_state()
        action = available_actions[individual.predict(state)]
        episode_reward += env.perform_action(action)
    return episode_reward """

if __name__ == '__main__':
    from building.discrete_floor_transition import DiscreteFloorTransition
    from caller.continuous_random_call import ContinuousRandomCallCaller
    from controller import generate_available_actions

    random.seed(s.RANDOM_SEED)

    caller = ContinuousRandomCallCaller()
    building = DiscreteFloorTransition(caller)
    
    state_vec, _ = building.sample_state()
    input_dims = state_vec.shape[0]
    available_actions = generate_available_actions()

    # Initial guess
    w = ANN(input_dims, s.FC_LAYER_PARAMS, len(available_actions))

    for step in range(s.NUM_ITERATIONS):
        episode_rewards = np.zeros(s.POPULATION_SIZE)
        population = [ANN(input_dims, s.FC_LAYER_PARAMS, len(available_actions)) for _ in range(s.POPULATION_SIZE)]
        for j in range(s.POPULATION_SIZE):
            w_try = population[j] + w
            episode_rewards[j] = evaluate_individual(w_try, building, available_actions)
        
        standard_score = (episode_rewards - np.mean(episode_rewards)) / np.std(episode_rewards)
        for j in range(s.POPULATION_SIZE):
            w = w + population[j] * (standard_score[j] * s.LEARNING_RATE/(s.POPULATION_SIZE * s.NOISE_STANDARD_DEVIATION))

        if step % s.EVAL_INTERVAL == 0:
            avg_return = 0
            for _ in range(s.NUM_EVAL_EPISODES):
                avg_return += evaluate_individual(w, building, available_actions)
            avg_return /= s.NUM_EVAL_EPISODES
            print('{{"metric": "avg_return", "value": {}, "step": {}}}'.format(avg_return, step))
            sys.stdout.flush()
        
        if step > 0 and step % s.POLICY_SAVER_INTERVAL == 0:
            weights_dir = str(pathlib.Path(__file__).parent.absolute()) + "/weights/"
            np.save(weights_dir + "policy_{}.npy".format(step), w)