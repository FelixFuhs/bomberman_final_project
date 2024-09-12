import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import deque
from .config import LEARNING_RATE, ACTIONS  # Importing necessary hyperparameters from config

# Set up logging
logging.basicConfig(filename='agent.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# CNN for DQN
class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 17 * 17, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 5  # Adjusted for the new state representation

    self.q_network = DQN(input_channels, len(ACTIONS)).to(self.device)
    self.target_network = DQN(input_channels, len(ACTIONS)).to(self.device)

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

    self.q_network.train()
    self.target_network.eval()

    # Using LEARNING_RATE from config.py
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.steps = 0
    self.epsilon = 1.0
    self.scores = []
    self.losses = []
    self.game_counter = 0
    self.total_training_steps = 0  # Track total training steps

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
    """Convert the game state into a feature tensor for CNN."""
    if game_state is None:
        return None

    # Create channels for different game elements
    channels = []

    # Channel 1: Walls and crates
    field = game_state['field'].copy()
    channels.append(field)

    # Channel 2: Coins
    coin_map = np.zeros_like(field)
    for coin in game_state['coins']:
        coin_map[coin] = 1
    channels.append(coin_map)

    # Channel 3: Bombs
    bomb_map = np.zeros_like(field)
    for bomb in game_state['bombs']:
        bomb_map[bomb[0]] = bomb[1]  # bomb timer as value
    channels.append(bomb_map)

    # Channel 4: Agent position
    agent_map = np.zeros_like(field)
    agent_map[game_state['self'][3]] = 1
    channels.append(agent_map)

    # Channel 5: Other agents
    others_map = np.zeros_like(field)
    for other in game_state['others']:
        others_map[other[3]] = 1
    channels.append(others_map)

    # Stack channels
    state = np.stack(channels, axis=0).astype(np.float32)

    return state

def create_graphs(self):
    """Generate and save performance and loss graphs."""
    create_performance_graph(self)
    create_loss_graph(self)
    logger.info(f"Graphs created for game {self.game_counter}, Total steps: {self.total_training_steps}")

def create_performance_graph(self):
    """Create and save performance graph."""
    plt.figure(figsize=(16, 9))
    plt.plot(range(1, len(self.scores) + 1), self.scores, label='Scores', color='#808080')
    
    if (len(self.scores) >= 100):
        avg_scores = np.convolve(self.scores, np.ones(100) / 100, 'valid')
        plt.plot(range(100, len(self.scores) + 1), avg_scores, label='100-game MA', color='red')

    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title(f'Score over {len(self.scores)} Games')
    plt.legend()
    plt.grid(True)
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