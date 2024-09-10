import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Add this as a global variable at the top of your file
position_history = deque(maxlen=2000)  # Adjust maxlen as needed
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Q-learning parameters
BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.95
TARGET_UPDATE_FREQUENCY = 100  # Update target network every 100 steps
MAX_GRAD_NORM = 1.0  # For gradient clipping

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(10, 128)  # 10 features as input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(ACTIONS))  # 6 possible actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for all actions

def setup(self):
    """
    Setup your model. This is called once when loading the agent.
    If there is a pre-trained model, it loads it, otherwise it initializes a new one.
    """
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize main and target networks
    self.q_network = DQN().to(self.device)
    self.target_network = DQN().to(self.device)

    # Load existing model if available, otherwise initialize
    if os.path.exists("dqn_model.pt"):
        self.q_network.load_state_dict(torch.load("dqn_model.pt", map_location=self.device, weights_only=True))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print("DQN model loaded successfully.")
    else:
        print("No pre-trained model found, initializing new model.")

    # Set the networks to evaluation mode by default
    self.q_network.eval()
    self.target_network.eval()

    # Optimizer
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)

    # Set step counter for target network updates
    self.steps = 0

def act(self, game_state: dict) -> str:
    """
    The agent selects an action based on the game state using epsilon-greedy policy during training,
    and always chooses the best action (based on the DQN model) during evaluation.
    """
    # Convert the game state into a feature vector
    features = state_to_features(game_state)
    if features is None:
        # If the features are None (e.g., game not started or invalid state), return a random action
        return np.random.choice(ACTIONS)

    # Convert features to tensor and pass through the DQN model
    features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
    
    # Training mode: use epsilon-greedy for exploration
    if self.train:
        if random.random() < self.epsilon:
            # With probability epsilon, choose a random action (explore)
            action = np.random.choice(ACTIONS)
        else:
            # Otherwise, choose the best action from the DQN (exploit)
            with torch.no_grad():
                q_values = self.q_network(features)
                action = ACTIONS[torch.argmax(q_values).item()]
        
        return action

    # Evaluation mode (not training): always choose the best action
    else:
        with torch.no_grad():
            q_values = self.q_network(features)
            return ACTIONS[torch.argmax(q_values).item()]

def state_to_features(game_state: dict) -> np.ndarray:
    """
    Converts the game state into a feature vector.
    """
    global position_history

    if game_state is None:
        position_history.clear()
        return None

    # Extract relevant information from game state
    arena = game_state['field']
    coins = game_state['coins']
    (name, score, bomb_possible, (x, y)) = game_state['self']

    # Initialize feature list
    features = []

    # Feature 1: Distance to nearest coin
    if coins:
        distances = [manhattan_distance((x, y), coin) for coin in coins]
        nearest_coin_distance = min(distances)
        features.append(nearest_coin_distance)
    else:
        features.append(-1)  # No coins left

    # Feature 2-3: Direction to nearest coin
    if coins:
        nearest_coin = coins[distances.index(nearest_coin_distance)]
        coin_direction = (nearest_coin[0] - x, nearest_coin[1] - y)
        features.extend(coin_direction)
    else:
        features.extend((0, 0))

    # Feature 4: Number of coins left
    features.append(len(coins))

    # Feature 5-8: Can move in each direction
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # up, right, down, left
        features.append(int(arena[x+dx, y+dy] == 0))

    # Update position history
    position_history.append((x, y))

    # Feature 9: Steps in current position
    steps_in_place = 1
    for past_pos in reversed(list(position_history)[:-1]):
        if past_pos == (x, y):
            steps_in_place += 1
        else:
            break
    features.append(steps_in_place)

    # Feature 10: Exploration ratio
    unique_positions = len(set(position_history))
    total_steps = len(position_history)
    exploration_ratio = unique_positions / total_steps if total_steps > 0 else 0
    features.append(exploration_ratio)

    return np.array(features, dtype=np.float32)

def manhattan_distance(pos1, pos2):
    return abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1]))