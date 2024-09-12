# Actions the agent can take
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Neural Network parameters
INPUT_CHANNELS = 5  # Number of input channels for the CNN (field, coins, bombs, agent, others)
HIDDEN_CHANNELS = [32, 64, 64]  # Number of channels in each convolutional layer

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
TARGET_UPDATE = 500  # Number of steps between target network updates
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 500  # Number of steps for epsilon decay

# Experience replay parameters
TRANSITION_HISTORY_SIZE = 100_000  # Size of the replay buffer

# Training loop parameters
TRAIN_EVERY_N_STEPS = 4  # How often (in steps) to train the network
TRAIN_ITERATIONS = 1  # How many batches to train each time

# Evaluation parameters
EVAL_EVERY_N_GAMES = 100  # How often (in games) to run evaluation

# Saving and logging
SAVE_MODEL_EVERY_N_GAMES = 500  # How often (in games) to save the model
LOG_EVERY_N_GAMES = 100  # How often (in games) to log detailed statistics

# CNN-specific parameters
CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
CONV_PADDING = 1

# Fully connected layers sizes
FC_SIZES = [512, 128]

# Game-specific parameters
FIELD_WIDTH = 17
FIELD_HEIGHT = 17
MAX_STEPS = 400  # Maximum number of steps in a game

# Reward shaping
REWARD_COIN_COLLECTED = 10
REWARD_KILLED_OPPONENT = 5
REWARD_INVALID_ACTION = -2
REWARD_WAITED = -1
REWARD_KILLED_SELF = -10
REWARD_SURVIVED_ROUND = 0.1
REWARD_CRATE_DESTROYED = 0.5
REWARD_COIN_FOUND = 1
REWARD_BOMB_DROPPED = -0.1
REWARD_MOVED = -0.01  # Small penalty for any movement to encourage efficiency