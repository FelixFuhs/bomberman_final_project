import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import deque

# Set up logging
logging.basicConfig(filename='agent.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 3  # Adjusted to match the feature vector size

    self.q_network = DQN(input_size, len(ACTIONS)).to(self.device)
    self.target_network = DQN(input_size, len(ACTIONS)).to(self.device)

    if os.path.exists("dqn_model.pt"):
        try:
            state_dict = torch.load("dqn_model.pt", map_location=self.device, weights_only=True)  # Added weights_only=True
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
            logger.info("DQN model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Initializing new model.")
    else:
        logger.info("No pre-trained model found, initializing new model.")

    self.q_network.eval()
    self.target_network.eval()
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-5)
    self.steps = 0
    self.epsilon = 1.0
    self.scores = []
    self.losses = []
    self.game_counter = 0


def act(self, game_state: dict) -> str:
    """Choose an action based on Q-values or random exploration."""
    features = state_to_features(game_state)
    if features is None:
        return np.random.choice(ACTIONS)

    features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

    with torch.no_grad():
        q_values = self.q_network(features)
        self.logger.debug(f"Q-values: {q_values.squeeze().tolist()}")  # Log Q-values

    if self.train and random.random() < self.epsilon:
        action = np.random.choice(ACTIONS)
    else:
        action = ACTIONS[torch.argmax(q_values).item()]
    
    return action


def state_to_features(game_state: dict) -> np.ndarray:
    """Convert the game state into the feature vector."""
    if game_state is None:
        return None

    # Example of minimal feature extraction
    coins = game_state['coins']
    (x, y) = game_state['self'][3]
    nearest_coin_dist = min([abs(x - cx) + abs(y - cy) for (cx, cy) in coins]) if coins else -1
    
    # Concatenating other example features
    features = [
        nearest_coin_dist, 
        game_state['self'][2],  # Bomb availability
        game_state['step'] / 400  # Normalized step number
    ]
    return np.array(features, dtype=np.float32)

def create_graphs(self):
    """Generate and save performance and loss graphs."""
    create_performance_graph(self)
    create_loss_graph(self)
    logger.info(f"Graphs created for game {self.game_counter}")

def create_performance_graph(self):
    """Create and save performance graph."""
    plt.figure(figsize=(16, 9))
    plt.plot(range(1, len(self.scores) + 1), self.scores, label='Scores', color='#808080')
    
    if len(self.scores) >= 100:
        avg_scores = np.convolve(self.scores, np.ones(100) / 100, 'valid')
        plt.plot(range(100, len(self.scores) + 1), avg_scores, label='100-game MA', color='red')

    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title(f'Score over {len(self.scores)} Games')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'performance_graph_{len(self.scores)}.png')
    plt.close()

def create_loss_graph(self):
    """Create and save loss graph."""
    if len(self.losses) > 0:
        plt.figure(figsize=(16, 9))
        plt.plot(range(1, len(self.losses) + 1), self.losses, label='Losses', color='red')

        if len(self.losses) > 10:
            avg_loss = np.convolve(self.losses, np.ones(10) / 10, 'valid')
            plt.plot(range(10, len(self.losses) + 1), avg_loss, label='10-step MA', color='blue')

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss over time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_graph_{len(self.losses)}.png')
        plt.close()
