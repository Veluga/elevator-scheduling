import numpy as np
import settings as s
import random
import pathlib
import sys
from enum import Enum

class ANN:
    """Hosts data of individuals.
    Consists of one or more network layers.
    """
    def __init__(self, input_dims, hidden_layer_units, num_actions, scale=s.NOISE_STANDARD_DEVIATION):
        # Input is extended by bias
        self.input_dims = input_dims
        self.hidden_layer_units = hidden_layer_units
        self.num_actions = num_actions

        # Generate random network layers
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
        """ReLu activation"""
        return np.maximum(x, 0)

    def _sigmoid(self, z):
        """Sigmoid activation"""
        return 1/(1+np.exp(-z))

    def predict(self, input_):
        """Forward propagation without backpropagation"""
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
        """ANN + ANN
        Adds layer matrices.
        """
        res = ANN(self.input_dims, self.hidden_layer_units, self.num_actions)
        res.input_layer = self.input_layer + other.input_layer
        
        for idx, (l, other_l) in enumerate(zip(self.hidden_layers, other.hidden_layers)):
            res.hidden_layers[idx] = l + other_l

        res.output_layer = self.output_layer + other.output_layer

        return res

    def __mul__(self, scalar):
        """ANN * scalar
        Scales layer matrices by scalar.
        """
        res = ANN(self.input_dims, self.hidden_layer_units, self.num_actions)
        res.input_layer = self.input_layer * scalar
        
        for idx, hl in enumerate(self.hidden_layers):
            res.hidden_layers[idx] = hl * scalar
        
        res.output_layer = self.output_layer * scalar
        
        return res

def evaluate_individual(individual, building, available_actions):
    """Gauge performance of individual ANN in building environment."""
    building.reset()
    episode_reward = 0
    for _ in range(s.EPISODE_LENGTH):
        state_vec, _ = building.sample_state()
        action = available_actions[individual.predict(state_vec)]
        episode_reward += sum(building.perform_action(action))
    return episode_reward

if __name__ == '__main__':
    from building.discrete_floor_transition import DiscreteFloorTransition
    from caller.get_caller import get_caller
    from benchmark_controller import generate_available_actions

    random.seed(s.RANDOM_SEED)
    np.random.seed(s.RANDOM_SEED)

    caller = get_caller()
    building = DiscreteFloorTransition(caller)
    
    state_vec, _ = building.sample_state()
    input_dims = state_vec.shape[0]
    available_actions = generate_available_actions()

    # Initial guess
    w = ANN(input_dims, s.FC_LAYER_PARAMS, len(available_actions))

    for step in range(s.NUM_ITERATIONS):
        episode_rewards = np.zeros(s.POPULATION_SIZE)
        # Generate random population
        population = [ANN(input_dims, s.FC_LAYER_PARAMS, len(available_actions)) for _ in range(s.POPULATION_SIZE)]
        for j in range(s.POPULATION_SIZE):
            # Evaluate individuals
            w_try = population[j] + w
            episode_rewards[j] = evaluate_individual(w_try, building, available_actions)
        
        # Calculate standard scores
        standard_score = (episode_rewards - np.mean(episode_rewards)) / np.std(episode_rewards)
        for j in range(s.POPULATION_SIZE):
            # Modify w proportional to individual's relative performance
            w = w + population[j] * (standard_score[j] * s.ES_LEARNING_RATE/(s.POPULATION_SIZE * s.NOISE_STANDARD_DEVIATION))

        if step % s.ES_EVAL_INTERVAL == 0:
            avg_return = 0
            for _ in range(s.ES_NUM_EVAL_EPISODES):
                avg_return += evaluate_individual(w, building, available_actions)
            avg_return /= s.ES_NUM_EVAL_EPISODES
            print('{{"metric": "avg_return", "value": {}, "step": {}}}'.format(avg_return, step))
            sys.stdout.flush()
        
        if step > 0 and step % s.ES_POLICY_SAVER_INTERVAL == 0:
            weights_dir = str(pathlib.Path(__file__).parent.absolute()) + "/weights/"
            np.save(weights_dir + "policy_{}.npy".format(step), w)