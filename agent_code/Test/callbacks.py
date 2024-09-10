import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Global variables
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
position_history = deque(maxlen=2000)  # Adjust maxlen as needed

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the input size based on the number of features
    input_size = 21  # Updated to match the new number of features in state_to_features

    # Initialize main and target networks
    self.q_network = DQN(input_size, len(ACTIONS)).to(self.device)
    self.target_network = DQN(input_size, len(ACTIONS)).to(self.device)

    # Load existing model if available and compatible, otherwise initialize new model
    if os.path.exists("dqn_model.pt"):
        try:
            state_dict = torch.load("dqn_model.pt", map_location=self.device, weights_only=True)
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
            print("DQN model loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Initializing new model with current architecture.")
    else:
        print("No pre-trained model found, initializing new model.")

    # Set the networks to evaluation mode by default
    self.q_network.eval()
    self.target_network.eval()

    # Optimizer
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)

    # Set step counter for target network updates
    self.steps = 0

    # Initialize epsilon for epsilon-greedy strategy
    self.epsilon = 1.0

    # Additional agent state
    self.steps_since_bomb = 0

    # Performance tracking
    self.game_counter = 0
    self.scores = []
    self.average_scores = []

    # Initialize losses tracking
    self.losses = []  # Track loss values

def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    # (existing end-of-round logic)

    # Performance graph update
    if self.game_counter % 100 == 0:
        self.create_performance_graph()

    # Loss graph update
    if len(self.losses) > 0 and self.game_counter % 100 == 0:
        self.create_loss_graph()



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param game_state: A dictionary describing the current game board.
    :return: A string representing an action
    """
    # Convert the game state into a feature vector
    features = state_to_features(game_state)
    if features is None:
        self.logger.warning("Features are None, choosing random action")
        return np.random.choice(ACTIONS)

    # Convert features to tensor and pass through the DQN model
    features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
    
    # Training mode: use epsilon-greedy for exploration
    if self.train:
        self.logger.debug(f"Training mode, current epsilon: {self.epsilon:.4f}")
        if random.random() < self.epsilon:
            action = np.random.choice(ACTIONS)
            self.logger.info(f"Chose random action due to epsilon: {action}")
        else:
            with torch.no_grad():
                q_values = self.q_network(features)
                action = ACTIONS[torch.argmax(q_values).item()]
            self.logger.info(f"Chose action based on Q-values: {action}")
            self.logger.debug(f"Q-values: {q_values.squeeze().tolist()}")
        
        # Decay epsilon
        old_epsilon = self.epsilon
        self.epsilon = max(0.05, self.epsilon * 0.995)
        self.logger.debug(f"Epsilon decayed from {old_epsilon:.4f} to {self.epsilon:.4f}")
        
        return action

    # Evaluation mode: always choose the best action
    else:
        self.logger.debug("Evaluation mode, choosing best action")
        with torch.no_grad():
            q_values = self.q_network(features)
            action = ACTIONS[torch.argmax(q_values).item()]
        self.logger.info(f"Chose action in evaluation mode: {action}")
        self.logger.debug(f"Q-values: {q_values.squeeze().tolist()}")
        return action

def state_to_features(game_state: dict) -> np.ndarray:
    """
    Converts the game state to a feature vector.
    """
    global position_history

    if game_state is None:
        position_history.clear()
        return None

    # Extract relevant information from game state
    arena = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    (x, y) = game_state['self'][3]
    
    # Initialize feature list
    features = []

    # Feature 1-3: Distance and direction to nearest coin (3 features)
    if coins:
        distances = [manhattan_distance((x, y), coin) for coin in coins]
        nearest_coin_distance = min(distances)
        features.append(nearest_coin_distance)
        nearest_coin = coins[distances.index(nearest_coin_distance)]
        features.extend(normalize_direction(nearest_coin[0] - x, nearest_coin[1] - y))
    else:
        features.extend([-1, 0, 0])  # No coins left

    # Feature 4-7: Danger levels in each direction (4 features)
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # up, right, down, left
        features.append(calculate_danger(arena, bombs, explosion_map, x+dx, y+dy))

    # Feature 8: Number of surrounding crates (1 feature)
    features.append(count_surrounding_crates(arena, x, y))

    # Feature 9: Bomb availability (1 feature)
    features.append(int(game_state['self'][2]))

    # Feature 10: Steps since last bomb placement (1 feature)
    features.append(game_state.get('steps_since_bomb', 0))

    # Feature 11-13: Nearest opponent distance and direction (3 features)
    opponent_info = get_nearest_opponent_info(game_state)
    features.extend(opponent_info)

    # Feature 14: Exploration ratio (1 feature)
    position_history.append((x, y))
    unique_positions = len(set(position_history))
    total_steps = len(position_history)
    exploration_ratio = unique_positions / total_steps if total_steps > 0 else 0
    features.append(exploration_ratio)

    # Feature 15: Steps in current position (1 feature)
    steps_in_place = 1
    for past_pos in reversed(list(position_history)[:-1]):
        if past_pos == (x, y):
            steps_in_place += 1
        else:
            break
    features.append(steps_in_place)

    # Feature 16: Number of coins left (1 feature)
    features.append(len(coins))

    # Feature 17: Current step number (normalized) (1 feature)
    features.append(game_state['step'] / 400)  # Assuming max steps is 400

    # New features 18-21: Path blocked in each direction (4 features)
    others = [agent[3] for agent in game_state['others']]
    bomb_xys = [bomb[0] for bomb in bombs]
    
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles = []
    for d in directions:
        if ((arena[d] == 0) and
            (explosion_map[d] < 1) and
            (d not in others) and
            (d not in bomb_xys)):
            valid_tiles.append(d)
    
    features.extend([
        int((x - 1, y) not in valid_tiles),  # LEFT
        int((x + 1, y) not in valid_tiles),  # RIGHT
        int((x, y - 1) not in valid_tiles),  # UP
        int((x, y + 1) not in valid_tiles)   # DOWN
    ])

    return np.array(features, dtype=np.float32)

def calculate_danger(arena, bombs, explosion_map, x, y):
    """
    Calculate the danger level at a given position.
    """
    if arena[x, y] == -1:  # Wall
        return 1
    danger = 0
    if explosion_map[x, y] > 0:
        danger = 1
    for (bx, by), t in bombs:
        if manhattan_distance((x, y), (bx, by)) <= 3:
            danger = max(danger, 1 - (t / 4))  # Normalize danger by bomb timer
    return danger

def count_surrounding_crates(arena, x, y):
    """
    Count the number of crates surrounding a given position.
    """
    return sum(arena[x+dx, y+dy] == 1 
               for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)])

def get_nearest_opponent_info(game_state):
    """
    Get information about the nearest opponent.
    """
    others = game_state['others']
    if not others:
        return [-1, 0, 0]  # No opponents
    (x, y) = game_state['self'][3]
    distances = [manhattan_distance((x, y), other[3]) for other in others]
    nearest_distance = min(distances)
    nearest_opponent = others[distances.index(nearest_distance)]
    direction = normalize_direction(nearest_opponent[3][0] - x, nearest_opponent[3][1] - y)
    return [nearest_distance] + direction

def normalize_direction(dx, dy):
    """
    Normalize a direction vector.
    """
    length = np.sqrt(dx**2 + dy**2)
    return [dx / length, dy / length] if length > 0 else [0, 0]

def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])




def create_performance_graph(self):
    """
    Creates and saves a graph of the average scores over time.
    """
    plt.figure(figsize=(16, 9))
    
    bg_color = '#f0f0f0'
    text_color = '#333333'
    score_color = '#808080'
    ma_100_color = '#ff0000'
    ma_250_color = '#ffa500'
    max_score_color = '#0000ff'
    grid_color = '#cccccc'

    plt.gca().set_facecolor(bg_color)
    plt.gcf().patch.set_facecolor(bg_color)

    plt.plot(range(1, self.game_counter + 1), self.scores, label='Individual Scores', alpha=0.5, color=score_color, linewidth=1)

    if self.game_counter >= 100:
        ma_100 = np.convolve(self.scores, np.ones(100), 'valid') / 100
        plt.plot(range(100, self.game_counter + 1), ma_100, label='100-game MA', color=ma_100_color, linewidth=3)

    if self.game_counter >= 250:
        ma_250 = np.convolve(self.scores, np.ones(250), 'valid') / 250
        plt.plot(range(250, self.game_counter + 1), ma_250, label='250-game MA', color=ma_250_color, linewidth=3)

    max_score = max(self.scores)
    max_index = self.scores.index(max_score)
    plt.scatter(max_index + 1, max_score, color=max_score_color, s=150, zorder=5, label=f'Max Score: {max_score}', edgecolors='black')

    plt.title(f'Agent Performance over {self.game_counter:,} Games', fontsize=20, fontweight='bold', color=text_color, pad=20)
    plt.xlabel('Game Number', fontsize=14, color=text_color, labelpad=10)
    plt.ylabel('Score', fontsize=14, color=text_color, labelpad=10)
    
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor=bg_color, edgecolor='none')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.grid(True, linestyle='--', alpha=0.7, color=grid_color, linewidth=0.8)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.annotate(f'Latest Score: {self.scores[-1]}', xy=(0.02, 0.06), xycoords='axes fraction', fontsize=12, color=text_color, fontweight='bold')
    
    if self.game_counter >= 100:
        plt.annotate(f'Latest 100-game MA: {ma_100[-1]:.2f}', xy=(0.02, 0.03), xycoords='axes fraction', fontsize=12, color=text_color, fontweight='bold')
    
    if self.game_counter >= 250:
        plt.annotate(f'Latest 250-game MA: {ma_250[-1]:.2f}', xy=(0.02, 0.00), xycoords='axes fraction', fontsize=12, color=text_color, fontweight='bold')

    plt.tick_params(colors=text_color, which='both', width=1, labelsize=10)

    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'performance_graph_{self.game_counter}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_loss_graph(self):
    """
    Creates and saves an improved graph of the loss over time.
    """
    plt.figure(figsize=(16, 9))
    
    bg_color = '#f0f0f0'
    text_color = '#333333'
    loss_color = '#ff0000'
    ma_color = '#0000ff'
    grid_color = '#cccccc'

    plt.gca().set_facecolor(bg_color)
    plt.gcf().patch.set_facecolor(bg_color)

    # Plot raw loss data
    plt.plot(range(1, len(self.losses) + 1), self.losses, label='Raw Loss', color=loss_color, alpha=0.3, linewidth=1)

    # Calculate and plot moving average
    window_size = min(100, len(self.losses) // 10)  # Adjust window size based on data length
    if window_size > 1:
        ma = np.convolve(self.losses, np.ones(window_size), 'valid') / window_size
        plt.plot(range(window_size, len(self.losses) + 1), ma, label=f'{window_size}-step Moving Average', color=ma_color, linewidth=2)

    plt.title(f'Loss Over Time (Games {self.game_counter-99} to {self.game_counter})', fontsize=20, fontweight='bold', color=text_color, pad=20)
    plt.xlabel('Training Steps', fontsize=14, color=text_color, labelpad=10)
    plt.ylabel('Loss (log scale)', fontsize=14, color=text_color, labelpad=10)
    
    plt.legend(fontsize=12, loc='upper right', frameon=True, facecolor=bg_color, edgecolor='none')

    plt.grid(True, linestyle='--', alpha=0.7, color=grid_color, linewidth=0.8)

    plt.yscale('log')  # Set y-axis to logarithmic scale

    plt.tick_params(colors=text_color, which='both', width=1, labelsize=10)

    plt.tight_layout()
    plt.savefig(f'loss_graph_{self.game_counter}.png', dpi=300, bbox_inches='tight')
    plt.close()
