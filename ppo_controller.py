from building.tf_building import TFBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from caller.interfloor_caller import InterfloorCaller
from caller.up_peak_caller import UpPeakCaller
from caller.down_peak_caller import DownPeakCaller
from caller.mixed_caller import MixedCaller
from benchmark_controller import generate_available_actions
from dqn_controller import compute_avg_return
import settings as s

from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common
from copy import deepcopy
import tensorflow as tf
import numpy as np
import pathlib
import random
import sys
import os
import time

# Code based on tf-agents documentation (https://github.com/tensorflow/agents/blob/29daabcc11b277a914f6b848d44c30b4aabbf659/tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py)

def collect_episode(environment, policy, num_episodes):
    # Collect episode trajectories to buffer
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1

def train():
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
        global_step = tf.compat.v1.train.get_or_create_global_step()

        actor_net = actor_distribution_network.ActorDistributionNetwork(
          train_env.observation_spec(),
          train_env.action_spec(),
          fc_layer_params=s.FC_LAYER_PARAMS,
          activation_fn=tf.keras.activations.tanh
        )
        value_net = value_network.ValueNetwork(
          train_env.observation_spec(),
          fc_layer_params=s.FC_LAYER_PARAMS,
          activation_fn=tf.keras.activations.tanh
        )
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=s.REINFORCE_LEARNING_RATE)

        agent = ppo_clip_agent.PPOClipAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            entropy_regularization=0.0,
            importance_ratio_clipping=0.2,
            normalize_observations=False,
            normalize_rewards=False,
            use_gae=True,
            num_epochs=s.NUM_ITERATIONS,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_step_counter=global_step
        )
        
        agent.initialize()

        # Saving config
        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            environment_steps_metric,
        ]
        train_metrics = step_metrics + [
            tf_metrics.AverageReturnMetric(
                batch_size=s.PPO_NUM_PARALLEL_ENVIRONMENTS),
            tf_metrics.AverageEpisodeLengthMetric(
                batch_size=s.PPO_NUM_PARALLEL_ENVIRONMENTS),
        ]

        root_dir = "/Users/albert/Desktop/Uni/Year 4/Thesis/src/ppo_data"
        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'eval')
        saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=1000
        )
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=1000
        )
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=s.PPO_NUM_EVAL_EPISODES),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=s.PPO_NUM_EVAL_EPISODES)
        ]

        # Replay buffer initialization
        eval_policy = agent.policy
        collect_policy = agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=s.PPO_NUM_PARALLEL_ENVIRONMENTS,
            max_length=s.PPO_REPLAY_BUFFER_CAPACITY
        )

        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics')
        )
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=eval_policy,
            global_step=global_step
        )
        saved_model = policy_saver.PolicySaver(
            eval_policy, train_step=global_step
        )

        train_checkpointer.initialize_or_restore()

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=s.PPO_COLLECT_EPISODES_PER_ITERATION
        )

        def train_step():
            trajectories = replay_buffer.gather_all()
            return agent.train(experience=trajectories)
        
        collect_time = 0
        train_time = 0
        timed_at_step = global_step.numpy()

        while environment_steps_metric.result() < s.NUM_ITERATIONS:
            global_step_val = global_step.numpy()
            if global_step_val > 0 and global_step_val % s.PPO_EVAL_INTERVAL == 0:
                start_time = time.time()
                metric_utils.eager_compute(
                    eval_metrics,
                    eval_env,
                    eval_policy,
                    num_episodes=s.PPO_NUM_EVAL_EPISODES,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )
                print("({}) The last eval step took me {} seconds!".format(global_step_val, time.time() - start_time))
                sys.stdout.flush()

            start_time = time.time()
            collect_driver.run()
            print("({}) The last driver run took me {} seconds!".format(global_step_val, time.time() - start_time))
            sys.stdout.flush()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            replay_buffer.clear()
            print("({}) The last training step took me {} seconds!".format(global_step_val, time.time() - start_time))
            sys.stdout.flush()
            train_time += time.time() - start_time

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=step_metrics
                )

            if global_step_val % s.PPO_LOG_INTERVAL == 0:
                print('step = %d, loss = %f', global_step_val, total_loss)
                steps_per_sec = (
                    (global_step_val - timed_at_step) / (collect_time + train_time)
                )
                print('%.3f steps/sec', steps_per_sec)
                print('collect_time = %.3f, train_time = %.3f', collect_time, train_time)
                sys.stdout.flush()
                with tf.compat.v2.summary.record_if(True):
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step
                    )

                if global_step_val % s.PPO_TRAIN_CHECKPOINT_INTERVAL == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % s.PPO_POLICY_CHECKPOINT_INTERVAL == 0:
                    policy_checkpointer.save(global_step=global_step_val)
                    saved_model_path = os.path.join(
                        saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9)
                    )
                    saved_model.save(saved_model_path)

                timed_at_step = global_step_val
                collect_time = 0
                train_time = 0


if __name__ == '__main__':
    random.seed(s.RANDOM_SEED)
    tf.random.set_seed(s.RANDOM_SEED)
    np.random.seed(s.RANDOM_SEED)

    train()