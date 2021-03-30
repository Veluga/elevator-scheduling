from building.tf_building import TFBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from caller.interfloor_caller import InterfloorCaller
from caller.up_peak_caller import UpPeakCaller
from caller.down_peak_caller import DownPeakCaller
from caller.mixed_caller import MixedCaller
from benchmark_controller import generate_available_actions
import settings as s

from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from copy import deepcopy
import numpy as np
import tensorflow as tf
import pathlib
import random
import sys


# Code based on tf-agents documentation (https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)

def collect_step(environment, policy, buffer):
    # Sample-act-sample(-add to buffer)
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

def compute_avg_return(environment, policy, num_episodes=10):
    """To gauge agent performance, calculate average return per episode over some number of episodes."""
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

if __name__ == '__main__':
    random.seed(s.RANDOM_SEED)
    tf.random.set_seed(s.RANDOM_SEED)
    np.random.seed(s.RANDOM_SEED)

    #with tf.device("/GPU:0"):
    with tf.device("/CPU:0"):
        tf.compat.v1.enable_v2_behavior()
        
        # Building initialization
        #caller = InterfloorCaller()
        #caller = UpPeakCaller()
        #caller = DownPeakCaller()
        caller = MixedCaller()
        train_py_building = TFBuilding(DiscreteFloorTransition(caller), generate_available_actions())
        eval_py_building = TFBuilding(DiscreteFloorTransition(caller), generate_available_actions())
        train_env = tf_py_environment.TFPyEnvironment(train_py_building)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_building)

        # Network and agent initialization
        q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=s.FC_LAYER_PARAMS
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=s.DQN_LEARNING_RATE)
        
        train_step_counter = tf.Variable(0)
        
        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
        )

        agent.initialize()
        agent.train = common.function(agent.train)

        # Replay buffer initialization
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=s.REPLAY_BUFFER_MAX_LENGTH
        )

        random_policy = random_tf_policy.RandomTFPolicy(
            train_env.time_step_spec(),
            train_env.action_spec()
        )
        
        # Collect a few trajectories using random policies to populate replay buffer
        collect_data(train_env, random_policy, replay_buffer, s.DQN_INITIAL_COLLECT_STEPS)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=s.BATCH_SIZE,
            num_steps=2
        ).prefetch(3)

        iterator = iter(dataset)

        # Enable continuous saving of policies
        policy_saver = PolicySaver(agent.policy, batch_size=None)
        collect_policy_saver = PolicySaver(agent.collect_policy, batch_size=None)
        weights_dir = str(pathlib.Path(__file__).parent.absolute()) + "/weights/"

        for _ in range(s.NUM_ITERATIONS):
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_data(train_env, agent.collect_policy, replay_buffer, s.DQN_COLLECT_STEPS_PER_ITERATION)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % s.DQN_LOG_INTERVAL == 0:
                env = train_env.envs[0].building
                waiting_passengers = sum([len(env.up_calls[f]) + len(env.down_calls[f]) for f in range(env.floors)])
                boarded_passengers = sum([len(e.buttons_pressed) for e in env.elevators])
                print('{{"metric": "loss", "value": {}, "step": {}}}'.format(train_loss, step))
                print('{{"metric": "waiting_passengers", "value": {}, "step": {}}}'.format(waiting_passengers, step))
                print('{{"metric": "boarded_passengers", "value": {}, "step": {}}}'.format(boarded_passengers, step))
                sys.stdout.flush()

            if step % s.DQN_EVAL_INTERVAL == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, s.DQN_NUM_EVAL_EPISODES)
                print('{{"metric": "avg_return", "value": {}, "step": {}}}'.format(avg_return, step))
                sys.stdout.flush()

            if step % s.DQN_POLICY_SAVER_INTERVAL == 0:
                policy_saver.save(weights_dir + 'policy_{}'.format(step))
                collect_policy_saver.save(weights_dir + 'collect_policy_{}'.format(step))