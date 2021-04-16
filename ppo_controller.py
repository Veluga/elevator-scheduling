from building.tensorforce_building import TensorforceBuilding
import settings as s


from tensorforce import Runner, Environment

def main():
    environment = Environment.create(
        environment=TensorforceBuilding, max_episode_timesteps=3600
    )

    # PPO agent specification
    agent = dict(
        agent='ppo',
        # Automatically configured network
        network='auto',
        # PPO optimization parameters
        batch_size=10, update_frequency=2, learning_rate=s.PPO_LEARNING_RATE, multi_step=10,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
        # Baseline network and optimizer
        baseline=dict(type='auto', size=s.FC_LAYER_PARAMS[0], depth=2),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory='model', frequency=10, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory='summaries', summaries=['loss', 'reward']),
        # Do not record agent-environment interaction trace
        recorder=None
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment, max_episode_timesteps=3600)

    runner.run(num_episodes=s.NUM_ITERATIONS)
    runner.close()


if __name__ == '__main__':
    main()