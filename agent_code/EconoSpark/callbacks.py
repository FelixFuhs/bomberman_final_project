# callbacks.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from collections import deque
from .config import LEARNING_RATE, ACTIONS, TEMPERATURE_START  # Import necessary hyperparameters from config

# Use the logger provided by the framework
logger = logging.getLogger(__name__)

# Neural Network for DQN with Batch Normalization and Dropout
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Neural network with increased capacity and Batch Normalization
        self.fc1 = nn.Linear(input_size, 600)   # First hidden layer with 600 neurons
        self.bn1 = nn.BatchNorm1d(600)          # BatchNorm for first layer
        self.fc2 = nn.Linear(600, 512)          # Second hidden layer with 512 neurons
        self.bn2 = nn.BatchNorm1d(512)          # BatchNorm for second layer
        self.fc3 = nn.Linear(512, 256)          # Third hidden layer with 256 neurons
        self.bn3 = nn.BatchNorm1d(256)          # BatchNorm for third layer
        self.fc4 = nn.Linear(256, 128)          # Fourth hidden layer with 128 neurons
        self.bn4 = nn.BatchNorm1d(128)          # BatchNorm for fourth layer
        self.fc5 = nn.Linear(128, 64)           # Fifth hidden layer with 64 neurons
        self.bn5 = nn.BatchNorm1d(64)           # BatchNorm for fifth layer
        self.fc6 = nn.Linear(64, 32)            # Sixth hidden layer with 32 neurons
        self.bn6 = nn.BatchNorm1d(32)           # BatchNorm for sixth layer
        self.fc7 = nn.Linear(32, output_size)   # Output layer

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.2)

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Pass data through the network with ReLU activations, BatchNorm, and Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.fc7(x)  # Output layer without activation
    


def setup(self):
    """Setup function to initialize networks and device."""
    # Detect and set the device
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
        device_name = f"GPU ({torch.cuda.get_device_name(self.device)})"
    else:
        self.device = torch.device("cpu")
        device_name = "CPU"

    print(f"Using device: {device_name}")  # Print statement to verify device
    self.logger.info(f"Model is using device: {device_name}")

    input_size = 19  # Adjusted to match the new feature vector size

    self.q_network = DQN(input_size, len(ACTIONS)).to(self.device)
    self.target_network = DQN(input_size, len(ACTIONS)).to(self.device)

    if os.path.exists("dqn_model.pt"):
        try:
            state_dict = torch.load("dqn_model.pt", map_location=self.device)
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
            self.logger.info("DQN model loaded successfully.")
        except Exception as exc:
            self.logger.error(f"Error loading model: {exc}. Initializing new model.")
    else:
        self.logger.info("No pre-trained model found, initializing new model.")

    self.q_network.eval()
    self.target_network.eval()

    # Using LEARNING_RATE from config.py
    # Added weight decay for regularization
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    self.steps = 0
    self.temperature = TEMPERATURE_START  # Initialize temperature
    self.scores = []
    self.losses = []
    self.game_counter = 0
    self.total_training_steps = 0  # Track total training steps

    # Initialize additional tracking metrics
    self.total_rewards = []
    self.positions_visited = set()
    self.coordinate_history = deque([], 15)  # Track the last 15 positions

def act(self, game_state: dict) -> str:
    """Choose an action based on Q-values."""
    features = state_to_features(self, game_state)
    if features is None:
        self.logger.debug("No features extracted, choosing a random action.")
        return np.random.choice(ACTIONS)

    features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

    with torch.no_grad():
        q_values = self.q_network(features).squeeze()
        self.logger.debug(f"Q-values: {q_values.tolist()}")  # Log Q-values
        self.logger.debug(f"Q-values stats - Mean: {q_values.mean().item()}, Max: {q_values.max().item()}, Min: {q_values.min().item()}")  # Log stats

    if self.train:
        # Subtract max Q-value to prevent numerical instability in softmax
        q_values_adjusted = q_values - q_values.max()
        # Apply temperature scaling to control exploration
        softmax_probs = torch.softmax(q_values_adjusted / self.temperature, dim=0).cpu().numpy()
        # Sample an action according to the softmax probabilities
        action_index = np.random.choice(len(ACTIONS), p=softmax_probs)
        self.logger.debug(f"Action probabilities: {softmax_probs}")
    else:
        # Select action with the highest Q-value during evaluation
        action_index = torch.argmax(q_values).item()

    action = ACTIONS[action_index]

    # Track the current position
    self.positions_visited.add(game_state['self'][3])
    # Add current position to the deque
    self.coordinate_history.append(game_state['self'][3])

    self.logger.debug(f"Chosen Action: {action}")
    return action


# Helper function for Manhattan distance
def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two positions."""
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

def is_move_valid(x, y, game_state):
    """Check if the move to (x, y) is valid based on obstacles, bombs, and explosions."""
    field = game_state['field']
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
        return False
    if field[x, y] != 0:  # Not a free tile (wall or crate)
        return False
    bombs = game_state['bombs']
    if any((bx, by) == (x, y) for (bx, by), _ in bombs):
        return False
    others = game_state['others']
    if any((ox, oy) == (x, y) for _, _, _, (ox, oy) in others):
        return False
    # Check for current explosions
    explosion_map = game_state['explosion_map']
    if explosion_map[x, y] > 0:
        return False
    return True

def count_crates_destroyed(arena, x, y):
    """Count the number of crates that would be destroyed by placing a bomb at position (x, y)."""
    count = 0
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        for i in range(1, 4):
            nx, ny = x + dx*i, y + dy*i
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if arena[nx, ny] == -1:  # Wall
                    break
                elif arena[nx, ny] == 1:  # Crate
                    count += 1
                    break  # Bomb blast stops at the first obstacle
                else:
                    continue
            else:
                break
    return count

def get_escape_routes(arena, x, y, bomb_map, game_state):
    """Calculate the number of safe directions the agent can move to after placing a bomb."""
    safe_directions = 0
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if is_move_valid(nx, ny, game_state):
            if bomb_map[nx, ny] > 0:
                safe_directions += 1
    return safe_directions

def state_to_features(self, game_state: dict) -> np.ndarray:
    """Convert the game state into a feature vector."""
    if game_state is None:
        return None

    # Extract basic information
    arena = game_state['field']
    x, y = game_state['self'][3]  # Agent's position
    bombs = game_state['bombs']
    bombs_left = game_state['self'][2]
    others = [xy for (_, _, _, xy) in game_state['others']]
    coins = game_state['coins']
    step = game_state['step']
    max_steps = 400  # Assuming maximum steps per game

    # Initialize features list
    features = []

    # Valid Moves
    directions = {'UP': (x, y - 1),
                  'DOWN': (x, y + 1),
                  'LEFT': (x - 1, y),
                  'RIGHT': (x + 1, y)}
    for move in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        nx, ny = directions[move]
        if is_move_valid(nx, ny, game_state):
            features.append(1)
        else:
            features.append(0)

    # Feature 5: Bomb Availability
    features.append(int(bombs_left > 0))

    # Feature 6-7: Proximity to Nearest Coin
    if coins:
        distances = [manhattan_distance((x, y), c) for c in coins]
        min_idx = np.argmin(distances)
        nearest_coin_dx = coins[min_idx][0] - x
        nearest_coin_dy = coins[min_idx][1] - y
        max_distance = arena.shape[0] + arena.shape[1]
        features.append(nearest_coin_dx / max_distance)
        features.append(nearest_coin_dy / max_distance)
    else:
        features.extend([0, 0])  # No coins left

    # Feature 8-9: Proximity to Nearest Crate
    crates = np.argwhere(arena == 1)
    if crates.size > 0:
        distances = [manhattan_distance((x, y), (cx, cy)) for (cx, cy) in crates]
        min_idx = np.argmin(distances)
        nearest_crate_dx = crates[min_idx][0] - x
        nearest_crate_dy = crates[min_idx][1] - y
        max_distance = arena.shape[0] + arena.shape[1]
        features.append(nearest_crate_dx / max_distance)
        features.append(nearest_crate_dy / max_distance)
    else:
        features.extend([0, 0])  # No crates left

    # Feature 10-11: Proximity to Nearest Opponent
    if others:
        distances = [manhattan_distance((x, y), o) for o in others]
        min_idx = np.argmin(distances)
        nearest_opponent_dx = others[min_idx][0] - x
        nearest_opponent_dy = others[min_idx][1] - y
        max_distance = arena.shape[0] + arena.shape[1]
        features.append(nearest_opponent_dx / max_distance)
        features.append(nearest_opponent_dy / max_distance)
    else:
        features.extend([0, 0])  # No opponents

    # Feature 12: Adjacent to Crate
    adjacent_crate = int(any(
        arena[x + dx, y + dy] == 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= x + dx < arena.shape[0] and 0 <= y + dy < arena.shape[1]
    ))
    features.append(adjacent_crate)

    # Feature 13: Adjacent to Opponent
    adjacent_opponent = int(any(
        (x + dx, y + dy) in others
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ))
    features.append(adjacent_opponent)

    # Feature 14: In Dead End
    walls = sum([
        1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if not (0 <= x + dx < arena.shape[0] and 0 <= y + dy < arena.shape[1]) or
           arena[x + dx, y + dy] == -1 or arena[x + dx, y + dy] == 1
    ])
    in_dead_end = int(walls >= 3)
    features.append(in_dead_end)

    # Corrected Bomb Map generation with explosion remaining dangerous for one more round
    bomb_map = np.ones(arena.shape) * 5

    # Include bombs and their future explosions
    for (bx, by), t in bombs:
        # The bomb position itself will be dangerous at t (when bomb explodes)
        explosion_time = t
        if 0 <= bx < arena.shape[0] and 0 <= by < arena.shape[1]:
            bomb_map[bx, by] = min(bomb_map[bx, by], explosion_time)

        # Spread the danger along the four cardinal directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for i in range(1, 4):  # Explosion extends up to 3 tiles
                nx, ny = bx + dx * i, by + dy * i
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    cell = arena[nx, ny]
                    if cell == -1:  # Stone wall blocks explosion
                        break
                    danger_time = t
                    bomb_map[nx, ny] = min(bomb_map[nx, ny], danger_time)
                    if cell == 1:  # Crate gets destroyed; explosion stops
                        break
                else:
                    break

    # Include currently active explosions from explosion_map
    explosions = game_state['explosion_map']  # 2D array with explosion timers
    for ix in range(arena.shape[0]):
        for iy in range(arena.shape[1]):
            if explosions[ix, iy] > 0:
                # The explosion remains dangerous this turn
                bomb_map[ix, iy] = -1  # Negative time indicates active explosion

    # Include positions that will be dangerous due to lingering explosions
    # After an explosion, the position remains dangerous for one more turn
    # Suppose we have a data structure listing positions with lingering explosions (if available)

    # Feature 15: In Bomb Blast Zone
    in_bomb_blast_zone = int(bomb_map[x, y] <= 0)
    # Feature 16: Bomb Timer (normalized)
    if bomb_map[x, y] <= 0:
        bomb_timer = (4 - abs(bomb_map[x, y])) / 4  # Normalize between 0 and 1
    elif bomb_map[x, y] < 5:
        bomb_timer = (bomb_map[x, y]) / 4  # Normalize between 0 and 1
    else:
        bomb_timer = 0
    features.append(in_bomb_blast_zone)
    features.append(bomb_timer)

    # Feature 17: Escape Routes (normalized)
    escape_routes = sum([
        is_move_valid(x + dx, y + dy, game_state)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ]) / 4.0
    features.append(escape_routes)

    # Feature 18: Loop Detection
    potential_loop = int(self.coordinate_history.count((x, y)) > 2)
    features.append(potential_loop)

    # Feature 19: Normalized Step
    normalized_step = step / max_steps
    features.append(normalized_step)

    # Combine all features into a single feature vector
    return np.array(features, dtype=np.float32)

def create_graphs(self):
    """Generate and save performance and loss graphs."""
    create_performance_graph(self)
    create_loss_graph(self)
    create_total_rewards_graph(self)
    self.logger.info(f"Graphs created for game {self.game_counter}, Total steps: {self.total_training_steps}")

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