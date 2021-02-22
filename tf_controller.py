from building.tf_building import TFBuilding
from building.building import ElevatorState
from building.discrete_floor_transition import DiscreteFloorTransition
from building.crites_barto import CritesBartoBuilding
from caller.continuous_random_call import ContinuousRandomCallCaller

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
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import tensorflow as tf


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

def collect_step(environment, policy, buffer):
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
    with tf.device("/GPU:0"):
        tf.compat.v1.enable_v2_behavior()
        
        caller = ContinuousRandomCallCaller()

        train_py_building = TFBuilding(CritesBartoBuilding(caller), ElevatorState) 
        eval_py_building = TFBuilding(CritesBartoBuilding(caller), ElevatorState)
        train_tf_building = tf_py_environment.TFPyEnvironment(train_py_building)
        eval_tf_building = tf_py_environment.TFPyEnvironment(eval_py_building)

        num_iterations = 20000 # @param {type:"integer"}

        initial_collect_steps = 100  # @param {type:"integer"}
        collect_steps_per_iteration = 1  # @param {type:"integer"}
        replay_buffer_max_length = 100000  # @param {type:"integer"}

        batch_size = 64  # @param {type:"integer"}
        learning_rate = 1e-5  # @param {type:"number"}
        log_interval = 100  # @param {type:"integer"}

        num_eval_episodes = 3  # @param {type:"integer"}
        eval_interval = 50000  # @param {type:"integer"}

        action_tensor_spec = tensor_spec.from_spec(train_tf_building.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # it's output.
        """ dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer]) """

        fc_layer_params = (50,50)
        q_net = q_network.QNetwork(
            train_tf_building.observation_spec(),
            train_tf_building.action_spec(),
            fc_layer_params=fc_layer_params
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            train_tf_building.time_step_spec(),
            train_tf_building.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_tf_building.batch_size,
            max_length=replay_buffer_max_length
        )

        agent.train = common.function(agent.train)

        # Reset the train step
        agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(eval_tf_building, agent.policy, num_eval_episodes)
        returns = [avg_return]

        random_policy = random_tf_policy.RandomTFPolicy(
            train_tf_building.time_step_spec(),
            train_tf_building.action_spec()
        )
        collect_data(train_tf_building, random_policy, replay_buffer, initial_collect_steps)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=batch_size, 
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        for _ in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_data(train_tf_building, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('{{"metric": "loss", "value": {}, "step": {}}}'.format(train_loss, step))
                #print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_tf_building, agent.policy, num_eval_episodes)
                print('{{"metric": "avg_return", "value": {}, "step": {}}}'.format(avg_return, step))
                returns.append(avg_return)