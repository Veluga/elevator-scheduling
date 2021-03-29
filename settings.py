# Simulation config
TICK_LENGTH_IN_SECONDS = 3
AVERAGE_CALL_FREQUENCY = 1
EPISODE_LENGTH = 3600
RANDOM_SEED = 0xDEADBEEF
REWARD_DELIVERED_PASSENGER = 2

# Building config
NUM_FLOORS = 10
NUM_ELEVATORS = 4
ELEVATOR_MAX_CAPACITY = 6

# General agent config
NUM_ITERATIONS = 100000000
REPLAY_BUFFER_MAX_LENGTH = 20000
BATCH_SIZE = 64
FC_LAYER_PARAMS = (100, 100)
EPSILON = 0.05
STEP_SIZE = 0.1
DISCOUNT_RATE = 0.99

# DQN config
DQN_INITIAL_COLLECT_STEPS = 5000
DQN_COLLECT_STEPS_PER_ITERATION = 1
DQN_LEARNING_RATE = 1e-5
DQN_LOG_INTERVAL = 10000
DQN_NUM_EVAL_EPISODES = 1
DQN_EVAL_INTERVAL = 50000
DQN_POLICY_SAVER_INTERVAL = DQN_EVAL_INTERVAL

# Categorical DQN agent config
NUM_ATOMS = 51
MIN_Q_VALUE = 0
MAX_Q_VALUE = NUM_ELEVATORS * REWARD_DELIVERED_PASSENGER
N_STEP_UPDATE = 2
CDQN_LEARNING_RATE = 1e-3
CDQN_LOG_INTERVAL = 5000
CDQN_EVAL_INTERVAL = 25000
CDQN_POLICY_SAVER_INTERVAL = CDQN_EVAL_INTERVAL

# REINFORCE agent config
REINFORCE_INITIAL_COLLECT_STEPS = 0
REINFORCE_COLLECT_EPISODES_PER_ITERATION = 2
REINFORCE_LOG_INTERVAL = 3
REINFORCE_NUM_EVAL_EPISODES = 3
REINFORCE_EVAL_INTERVAL = 15
REINFORCE_LEARNING_RATE = 1e-3
REINFORCE_POLICY_SAVER_INTERVAL = REINFORCE_EVAL_INTERVAL

# ES config
ES_LEARNING_RATE = 1e-2
ES_LOG_INTERVAL = 10000
ES_NUM_EVAL_EPISODES = 3
ES_EVAL_INTERVAL = 3
ES_POLICY_SAVER_INTERVAL = ES_EVAL_INTERVAL

POPULATION_SIZE = 50
NOISE_STANDARD_DEVIATION = 0.1