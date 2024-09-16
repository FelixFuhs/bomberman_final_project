import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from collections import deque
from .config import LEARNING_RATE, ACTIONS, TEMPERATURE_START  # Import necessary hyperparameters from config

# Set up logging
logging.basicConfig(filename='agent.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Neural network with increased capacity
        self.fc1 = nn.Linear(input_size, 512)   # First hidden layer with 512 neurons
        self.fc2 = nn.Linear(512, 256)          # Second hidden layer with 256 neurons
        self.fc3 = nn.Linear(256, 128)          # Third hidden layer with 128 neurons
        self.fc4 = nn.Linear(128, 64)           # Fourth hidden layer with 64 neurons
        self.fc5 = nn.Linear(64, 32)            # Fifth hidden layer with 32 neurons
        self.fc6 = nn.Linear(32, output_size)   # Output layer

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Pass data through the network with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)  # Output layer without activation

def setup(self):
    #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = "cpu"
    input_size = 21  # Adjusted to match the new feature vector size

    self.q_network = DQN(input_size, len(ACTIONS)).to(self.device)
    self.target_network = DQN(input_size, len(ACTIONS)).to(self.device)

    if os.path.exists("dqn_model.pt"):
        try:
            state_dict = torch.load("dqn_model.pt", map_location=self.device)
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
            logger.info("DQN model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Initializing new model.")
    else:
        logger.info("No pre-trained model found, initializing new model.")

    self.q_network.eval()
    self.target_network.eval()

    # Using LEARNING_RATE from config.py
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.steps = 0
    self.temperature = TEMPERATURE_START  # Initialize temperature
    self.scores = []
    self.losses = []
    self.game_counter = 0
    self.total_training_steps = 0  # Track total training steps

    # Initialize additional tracking metrics
    self.total_rewards = []

def act(self, game_state: dict) -> str:
    """Choose an action based on Q-values."""
    features = state_to_features(game_state)
    if features is None:
        
        self.logger.debug(f"chose a random action")  # Log Q-values
        return np.random.choice(ACTIONS)

    features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

    with torch.no_grad():
        q_values = self.q_network(features).squeeze()
        self.logger.debug(f"Q-values: {q_values.tolist()}")  # Log Q-values

    if self.train:
        # Subtract max Q-value to prevent numerical instability in softmax
        q_values_adjusted = q_values - q_values.max()
        # Apply temperature scaling to control exploration
        softmax_probs = torch.softmax(q_values_adjusted / self.temperature, dim=0).cpu().numpy()
        # Sample an action according to the softmax probabilities
        action_index = np.random.choice(len(ACTIONS), p=softmax_probs)
    else:
        # Select action with the highest Q-value during evaluation
        action_index = torch.argmax(q_values).item()

    action = ACTIONS[action_index]

    # Track the current position
    if not hasattr(self, 'positions_visited'):
        self.positions_visited = set()
    self.positions_visited.add(game_state['self'][3])

    # Add current position to the deque
    self.coordinate_history.append(game_state['self'][3])

    self.logger.debug(f"Chosen Action: " + action)  # Log Q-values
    return action

def is_move_valid(x, y, game_state):
    """Check if the move to (x, y) is valid based on obstacles and bombs."""
    field = game_state['field']
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
        return False
    if field[x, y] != 0:  # Not a free tile
        return False
    bombs = game_state['bombs']
    if any((bx, by) == (x, y) for (bx, by), _ in bombs):
        return False
    others = game_state['others']
    if any((ox, oy) == (x, y) for _, _, _, (ox, oy) in others):
        return False
    return True

def state_to_features(game_state: dict) -> np.ndarray:
    """Convert the game state into a feature vector."""
    if game_state is None:
        return None

    arena = game_state['field']
    (x, y) = game_state['self'][3]  # Agent's position
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = [xy for (_, _, _, xy) in game_state['others']]

    # Initialize bomb map
    bomb_map = np.ones(arena.shape) * 5  # Distance to the nearest bomb
    for (bx, by), t in bombs:
        for (i, j) in [(bx + h, by) for h in range(-3, 4)] + [(bx, by + h) for h in range(-3, 4)]:
            if 0 <= i < arena.shape[0] and 0 <= j < arena.shape[1]:
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Feature 1-4: Valid moves (UP, DOWN, LEFT, RIGHT)
    valid_up = is_move_valid(x, y - 1, game_state)
    valid_down = is_move_valid(x, y + 1, game_state)
    valid_left = is_move_valid(x - 1, y, game_state)
    valid_right = is_move_valid(x + 1, y, game_state)

    # Feature 5: Bomb availability (binary feature)
    bomb_available = int(game_state['self'][2])  # 1 if the agent can place a bomb, 0 otherwise

    # Feature 6-7: Nearest coin delta x and delta y (normalized)
    if coins:
        distances = [(cx - x, cy - y) for (cx, cy) in coins]
        nearest_coin_dx, nearest_coin_dy = min(distances, key=lambda d: abs(d[0]) + abs(d[1]))
        max_distance = arena.shape[0] + arena.shape[1]
        nearest_coin_dx /= max_distance
        nearest_coin_dy /= max_distance
    else:
        nearest_coin_dx = 0  # No coins left
        nearest_coin_dy = 0  # No coins left

    # Feature 8: Number of coins remaining (normalized)
    num_coins_remaining = len(coins) / ((arena.shape[0] - 2) * (arena.shape[1] - 2))

    # Feature 9: Proximity to nearest opponent (normalized)
    if others:
        nearest_opponent_dist = min([abs(x - ox) + abs(y - oy) for (ox, oy) in others]) / (arena.shape[0] + arena.shape[1])
    else:
        nearest_opponent_dist = -1  # No opponents

    # Feature 10: Proximity to bombs (normalized)
    if bombs:
        bomb_dists = [abs(x - bx) + abs(y - by) for (bx, by), _ in bombs]
        nearest_bomb_dist = min(bomb_dists) / (arena.shape[0] + arena.shape[1])
    else:
        nearest_bomb_dist = -1  # No bombs

    # Feature 11: In a dead end (binary feature)
    walls = [arena[x + dx, y + dy] == -1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    dead_end = int(sum(walls) >= 3)

    # Feature 12: Number of adjacent crates (normalized)
    adjacent_crates = sum(
        [arena[x + dx, y + dy] == 1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    ) / 4  # Normalize by maximum possible adjacent crates

    # Feature 13: Normalized step count (current step / max steps)
    normalized_step = game_state['step'] / 400

    # Feature 14: Bias feature (constant value)
    bias_feature = 1  # A constant feature to help the network learn bias

    # Additional Features

    # Feature 15: Number of valid actions (normalized)
    num_valid_actions = sum([valid_up, valid_down, valid_left, valid_right]) / 4.0

    # Feature 16: Number of crates left (normalized)
    num_crates_left = np.sum(arena == 1) / ((arena.shape[0] - 2) * (arena.shape[1] - 2))  # Exclude walls

    # Feature 17: Is agent adjacent to a bomb about to explode (binary)
    adjacent_positions = [(x + dx, y + dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]
    adjacent_bomb = int(any(bomb_map[pos] <= 3 for pos in adjacent_positions if 0 <= pos[0] < arena.shape[0] and 0 <= pos[1] < arena.shape[1]))

    # Feature 18: Time until explosion at agent's position (normalized)
    time_until_explosion = bomb_map[x, y] / 5.0  # Normalize by maximum bomb timer

    # Feature 19: Is agent adjacent to an opponent (binary)
    adjacent_opponent = int(any(pos in others for pos in adjacent_positions))

    # Feature 20: Agent's own position (normalized)
    agent_pos_x = x / arena.shape[0]
    agent_pos_y = y / arena.shape[1]

    # Combine all features into a single feature vector
    features = [
        valid_up, valid_down, valid_left, valid_right,
        bomb_available, nearest_coin_dx, nearest_coin_dy,
        num_coins_remaining, nearest_opponent_dist, nearest_bomb_dist,
        dead_end, adjacent_crates, normalized_step, bias_feature,
        num_valid_actions, num_crates_left, adjacent_bomb,
        time_until_explosion, adjacent_opponent, agent_pos_x, agent_pos_y
    ]

    return np.array(features, dtype=np.float32)

def create_graphs(self):
    """Generate and save performance and loss graphs."""
    create_performance_graph(self)
    create_loss_graph(self)
    create_total_rewards_graph(self)
    logger.info(f"Graphs created for game {self.game_counter}, Total steps: {self.total_training_steps}")

def create_performance_graph(self):
    """Create and save performance graph."""
    plt.figure(figsize=(16, 9))
    plt.plot(range(1, len(self.scores) + 1), self.scores, label='Scores', color='#808080')

    if len(self.scores) >= 100:
        avg_scores_100 = np.convolve(self.scores, np.ones(100) / 100, 'valid')
        plt.plot(range(100, len(self.scores) + 1), avg_scores_100, label='100-game MA', color='red')

    if len(self.scores) >= 250:
        avg_scores_250 = np.convolve(self.scores, np.ones(250) / 250, 'valid')
        plt.plot(range(250, len(self.scores) + 1), avg_scores_250, label='250-game MA', color='green')

    if len(self.scores) >= 500:
        avg_scores_500 = np.convolve(self.scores, np.ones(500) / 500, 'valid')
        plt.plot(range(500, len(self.scores) + 1), avg_scores_500, label='500-game MA', color='blue')

    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title(f'Score over {len(self.scores)} Games')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'performance_graph_{self.game_counter}.png')
    plt.close()

def create_loss_graph(self):
    """Create and save loss graph with logarithmic scale."""
    if len(self.losses) > 0:
        plt.figure(figsize=(16, 9))

        # Convert losses to numpy array and replace zeros/negatives with a small positive value
        losses = np.array(self.losses)
        min_positive = np.min(losses[losses > 0]) if np.any(losses > 0) else 1e-10
        losses[losses <= 0] = min_positive / 10  # Set to a value smaller than the smallest positive loss

        plt.semilogy(range(1, len(losses) + 1), losses, label='Losses', color='red')

        if len(losses) > 10:
            # Use convolution for moving average, handling log scale
            log_losses = np.log10(losses)
            avg_log_loss = np.convolve(log_losses, np.ones(10) / 10, 'valid')
            avg_loss = 10 ** avg_log_loss
            plt.semilogy(range(10, len(losses) + 1), avg_loss, label='10-step MA', color='blue')

        plt.xlabel('Training Steps')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss over time (logarithmic scale)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'loss_graph_{self.total_training_steps}.png')
        plt.close()

def create_total_rewards_graph(self):
    """Create and save total rewards graph."""
    if len(self.total_rewards) > 0:
        plt.figure(figsize=(16, 9))
        plt.plot(range(1, len(self.total_rewards) + 1), self.total_rewards, label='Total Rewards', color='green')

        if len(self.total_rewards) >= 100:
            avg_rewards_100 = np.convolve(self.total_rewards, np.ones(100) / 100, 'valid')
            plt.plot(range(100, len(self.total_rewards) + 1), avg_rewards_100, label='100-game MA', color='red')

        if len(self.total_rewards) >= 250:
            avg_rewards_250 = np.convolve(self.total_rewards, np.ones(250) / 250, 'valid')
            plt.plot(range(250, len(self.total_rewards) + 1), avg_rewards_250, label='250-game MA', color='blue')

        if len(self.total_rewards) >= 500:
            avg_rewards_500 = np.convolve(self.total_rewards, np.ones(500) / 500, 'valid')
            plt.plot(range(500, len(self.total_rewards) + 1), avg_rewards_500, label='500-game MA', color='purple')

        plt.xlabel('Game Number')
        plt.ylabel('Total Rewards')
        plt.title('Total Rewards over Games')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'total_rewards_graph_{self.game_counter}.png')
        plt.close()
