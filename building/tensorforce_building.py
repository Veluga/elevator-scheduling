from .discrete_floor_transition import DiscreteFloorTransition
from caller.get_caller import get_caller
from benchmark_controller import generate_available_actions
import settings as s

from tensorforce import Environment

class TensorforceBuilding(Environment):

    def __init__(self):
        super().__init__()
        caller = get_caller()
        self._building = DiscreteFloorTransition(caller)
        self._state, _ = self._building.sample_state()
        self._available_actions = generate_available_actions()
        self._timestep = 0

    def states(self):
        return dict(type='int', shape=self._state.shape, num_values=s.NUM_FLOORS)

    def actions(self):
        return dict(type='int', num_values=len(self._available_actions))

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return s.EPISODE_LENGTH

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        self._building.reset()
        self._state, _ = self._building.sample_state()
        self._timestep = 0
        return self._state

    def execute(self, actions):
        action = self._available_actions[actions]
        rewards = self._building.perform_action(action)
        self._timestep += 1
        self._state, _ = self._building.sample_state()
        terminal = self._timestep >= s.EPISODE_LENGTH
        return self._state, terminal, sum(rewards)
