from building.tf_building import TensorflowAgentsBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from caller.get_caller import get_caller
from benchmark_controller import generate_available_actions
from dqn_controller import collect_step, collect_data, compute_avg_return
import settings as s

from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.networks import categorical_q_network
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


# Code based on tf-agents documentation (https://www.tensorflow.org/agents/tutorials/9_c51_tutorial)

if __name__ == '__main__':
    random.seed(s.RANDOM_SEED)
    tf.random.set_seed(s.RANDOM_SEED)
    np.random.seed(s.RANDOM_SEED)

    #with tf.device("/GPU:0"):
    with tf.device("/CPU:0"):
        tf.compat.v1.enable_v2_behavior()
        
        # Building initialization
        caller = get_caller()
        train_py_building = TensorflowAgentsBuilding(DiscreteFloorTransition(caller), generate_available_actions())
        eval_py_building = TensorflowAgentsBuilding(DiscreteFloorTransition(caller), generate_available_actions())
        train_env = tf_py_environment.TFPyEnvironment(train_py_building)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_building)

        # Network and agent initialization
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            num_atoms=s.NUM_ATOMS,
            fc_layer_params=s.FC_LAYER_PARAMS
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=s.CDQN_LEARNING_RATE)
        
        train_step_counter = tf.compat.v2.Variable(0)
        
        agent = categorical_dqn_agent.CategoricalDqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            categorical_q_network=categorical_q_net,
            optimizer=optimizer,
            min_q_value=s.MIN_Q_VALUE,
            max_q_value=s.MAX_Q_VALUE,
            n_step_update=s.N_STEP_UPDATE,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=s.DISCOUNT_RATE,
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
            num_steps=s.N_STEP_UPDATE+1
        ).prefetch(3)

        iterator = iter(dataset)

        # Enable continuous saving of policies
        policy_saver = PolicySaver(agent.policy, batch_size=None)
        collect_policy_saver = PolicySaver(agent.collect_policy, batch_size=None)
        weights_dir = str(pathlib.Path(__file__).parent.absolute()) + "/weights/"

        for _ in range(s.NUM_ITERATIONS):
            # Collect a few steps using collect_policy and save to the replay buffer
            collect_data(train_env, agent.collect_policy, replay_buffer, s.DQN_COLLECT_STEPS_PER_ITERATION)

            # Sample a batch of data from the buffer and update the agent's network
            experience, _ = next(iterator)
            train_loss = agent.train(experience)

            step = agent.train_step_counter.numpy()

            if step % s.CDQN_LOG_INTERVAL == 0:
                env = train_env.envs[0].building
                waiting_passengers = sum([len(env.up_calls[f]) + len(env.down_calls[f]) for f in range(env.floors)])
                boarded_passengers = sum([len(e.buttons_pressed) for e in env.elevators])
                print('{{"metric": "loss", "value": {}, "step": {}}}'.format(train_loss.loss, step))
                print('{{"metric": "waiting_passengers", "value": {}, "step": {}}}'.format(waiting_passengers, step))
                print('{{"metric": "boarded_passengers", "value": {}, "step": {}}}'.format(boarded_passengers, step))
                sys.stdout.flush()

            if step % s.CDQN_EVAL_INTERVAL == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, s.DQN_NUM_EVAL_EPISODES)
                print('{{"metric": "avg_return", "value": {}, "step": {}}}'.format(avg_return, step))
                sys.stdout.flush()

            if step % s.CDQN_POLICY_SAVER_INTERVAL == 0:
                policy_saver.save(weights_dir + 'policy_{}'.format(step))
                collect_policy_saver.save(weights_dir + 'collect_policy_{}'.format(step))