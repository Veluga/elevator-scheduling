# Simulation config
TICK_LENGTH_IN_SECONDS = 1
AVERAGE_CALL_FREQUENCY = 5
EPISODE_LENGTH = 3600
RANDOM_SEED = 0xDEADBEEF

# Building config
NUM_FLOORS = 10
NUM_ELEVATORS = 4
FLOOR_HEIGHT = 4.0

# Non-TF Agent config
EPSILON = 0.05
STEP_SIZE = 0.1
DISCOUNT_RATE = 0.99

# Elevator config
MAX_VELOCITY=3.0
ACCELERATION=1.0
FLOOR_TIME=2
STOP_TIME=7
TURN_TIME=1

# Tensorflow config
NUM_ITERATIONS = 100000000
INTIIAL_COLLECT_STEPS = 10000
COLLECT_STEPS_PER_ITERATION = 1
REPLAY_BUFFER_MAX_LENGTH = 100000
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
LOG_INTERVAL = 10000
NUM_EVAL_EPISODES = 1
EVAL_INTERVAL = 50000
FC_LAYER_PARAMS = (100, 100, 100, 100)
POLICY_SAVER_INTERVAL = EVAL_INTERVAL