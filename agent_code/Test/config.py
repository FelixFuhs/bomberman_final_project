# config.py

# List of actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Other hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.005
EPSILON_DECAY = 20000
TRANSITION_HISTORY_SIZE = 100_000  # Max transitions stored in replay buffer
