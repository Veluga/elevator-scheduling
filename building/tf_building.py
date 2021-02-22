from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from enum import IntEnum, auto
from math import sqrt
import tensorflow as tf
import numpy as np
import settings as s


class TFBuilding(py_environment.PyEnvironment):
    def __init__(self, building, available_actions):
        self.building = building
        self.available_actions = available_actions
        self._timestep = 0
        self._episode_ended = False
        self._rewards = 0
        self._state = self._sample_state()
        # Tensorflow init
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=len(self.available_actions)-1,
            name='action'
        )
        self._observation_spec = array_spec.ArraySpec(
            shape=self._state.shape,
            dtype=np.int32,
            name='observation'
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _sample_state(self):
        state_vector, _ = self.building.sample_state()
        return state_vector

    def _reset(self):
        self._timestep = 0
        self._rewards = 0
        self._episode_ended = False
        self.building.reset()
        self._state = self._sample_state()
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action_idx):
        if self._episode_ended:
            return self._reset()

        self._timestep += 1

        """ if self._timestep % int(sqrt(1000)) == 0:
            print("Time {}/{}".format(self._timestep, 1000)) """

        reward = sum(self.building.perform_action(self.available_actions[action_idx]))

        self._state = self._sample_state()
        if self._timestep >= s.EPISODE_LENGTH:
            self._episode_ended = True
            return ts.termination(
                np.array(self._state, dtype=np.int32),
                reward=reward
            )
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=reward,
                discount=s.DISCOUNT_RATE
            )