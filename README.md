# About

An elevator scheduling simulation environment to be used for the development of (reinforcement) learning algorithms in Python.

Includes example agent implementations for:
* Tabular Q Learning agent (`agent/tabular_q_learning.py`)
* DQN agent (`dqn_controller.py`)
* Categorical DQN agent (`cdqn_controller.py`)
* Evolution Strategy agent (`es_controller.py`)
* REINFORCE (`reinforce_controller.py`)
* Heuristic benchmarks
  * FCFS Round Robin
  * Random Policy

Call generating policies implemented include:
* Pure up-peak traffic
* Pure down-peak traffic
* Pure interfloor traffic
* Mixed traffic (e.g. observed at lunch time in an office building)

# Architecture

* Agent
  * Given a building state representation, agent must respond with action to be executed for building (e.g. let elevator 1 ascend, elevator 2 stop, etc.)
  * Abstract class `Agent` (`agent/agent.py`) describes interface controllers can program against
* Building
  * Acts as the environment the agent operates in
  * Abstract class `Building` (`building/building.py`) describes interface controllers can program against
  * `building/tf_building.py` wraps `Building` environment as a `tf-agents PyEnvironment` for use in `tf-agents` agents
* Caller
  * Implements call generating policy of building
  * Abstract class `Caller` (`caller/caller.py`) describes interface `Building` can program against
* Controller
  * Ties together all moving parts of the system (i.e. agent, building (environment), caller)
  * For reinforcement learning agents, the controller implements the staple sample-act-sample-update loop
* Visualization
  * Allows metrics such as cumulative/average reward to be calculated and plotted after training
  * Abstract class `Visualization` (`visualization/visualization.py`) describes interface controller can program against
* `settings.py`
  * Contains global constants of the simulation environment and the learning agents (e.g. number of floors, number of elevators, etc.)