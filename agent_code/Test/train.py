import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import List
import events as e
from .callbacks import state_to_features, DQN, ACTIONS, MAX_GRAD_NORM

# Hyperparameters
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.95
TARGET_UPDATE_FREQUENCY = 100
REPLAY_BUFFER_CAPACITY = 10000
PRIORITIZED_REPLAY_ALPHA = 0.6
PRIORITIZED_REPLAY_BETA_START = 0.4
PRIORITIZED_REPLAY_BETA_FRAMES = 100000

# Epsilon parameters
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EPSILON_DECAY = 0.9995

# Custom events for reward shaping
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
EXPLORED_NEW_POSITION = "EXPLORED_NEW_POSITION"

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        if len(self.buffer) == 0:
            return None, None, None

        priorities = self.priorities[:self.size]
        probabilities = priorities ** PRIORITIZED_REPLAY_ALPHA
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size

def setup_training(self):
    """
    Initialize for training. This is called once per training session.
    """
    self.transitions = PrioritizedReplayBuffer(REPLAY_BUFFER_CAPACITY)

    # Set the device (GPU or CPU)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    self.q_network = DQN().to(self.device)
    self.target_network = DQN().to(self.device)

    # Load existing model if available
    try:
        self.q_network.load_state_dict(torch.load("dqn_model.pt", map_location=self.device, weights_only=True))
        print("Loaded existing DQN model.")
    except FileNotFoundError:
        print("No existing model found, training from scratch.")

    # Set the optimizer
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=INITIAL_LEARNING_RATE)
    
    # Learning rate scheduler
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)

    # Ensure target network starts with the same weights as q_network
    self.target_network.load_state_dict(self.q_network.state_dict())
    self.target_network.eval()

    # Set up performance tracking
    self.steps = 0
    self.game_counter = 0
    self.scores = []
    self.average_scores = []
    self.beta = PRIORITIZED_REPLAY_BETA_START
    
    # Initialize epsilon
    self.epsilon = INITIAL_EPSILON

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called each step to allow the agent to learn from the game events.
    """
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if old_features is None or new_features is None:
        return

    # Convert action to an index
    action_idx = ACTIONS.index(self_action)

    # Check if the agent has been killed or the game is over
    done = e.GOT_KILLED in events or new_game_state['step'] == 400

    # Calculate reward based on game events
    reward = reward_from_events(self, events)

    # Push this transition into the replay buffer
    self.transitions.push(old_features, action_idx, reward, new_features, done)

    # Train the network if enough transitions are stored
    if len(self.transitions) >= BATCH_SIZE:
        update_q_network(self)

    # Update the target network periodically
    self.steps += 1
    if self.steps % TARGET_UPDATE_FREQUENCY == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Update beta for prioritized replay
    self.beta = min(1.0, PRIORITIZED_REPLAY_BETA_START + self.steps * (1.0 - PRIORITIZED_REPLAY_BETA_START) / PRIORITIZED_REPLAY_BETA_FRAMES)

def update_q_network(self):
    """
    Perform one step of Q-learning to update the network using the prioritized replay buffer.
    """
    samples, indices, weights = self.transitions.sample(BATCH_SIZE, self.beta)
    if samples is None:
        return

    states, actions, rewards, next_states, dones = zip(*samples)

    # Convert batch data to tensors
    states = torch.from_numpy(np.array(states)).float().to(self.device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    
    # Handle None values in next_states
    non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
    non_final_next_states = torch.from_numpy(np.array([s for s in next_states if s is not None])).float().to(self.device)
    
    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
    weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

    # Get the predicted Q-values for the current states
    q_values = self.q_network(states).gather(1, actions).squeeze(1)

    # Compute the target Q-values
    next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
    
    # Compute the target Q-value
    targets = rewards + DISCOUNT_FACTOR * next_state_values * (1 - dones)

    # Compute the loss
    td_errors = torch.abs(q_values - targets).detach().cpu().numpy()
    loss = (weights * (q_values - targets) ** 2).mean()

    # Optimize the network
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), MAX_GRAD_NORM)
    self.optimizer.step()

    # Update priorities in the replay buffer
    self.transitions.update_priorities(indices, td_errors + 1e-6)  # Small constant to prevent zero priority

    # Step the learning rate scheduler
    self.scheduler.step()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each round to finalize learning for the round.
    """
    # Store the final transition
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    done = e.GOT_KILLED in events or last_game_state['step'] == 400

    if last_features is not None:
        action_idx = ACTIONS.index(last_action)
        self.transitions.push(last_features, action_idx, reward, None, done)

    # Save the model after every round
    torch.save(self.q_network.state_dict(), "dqn_model.pt")

    # Track scores and performance
    score = last_game_state['self'][1]
    self.scores.append(score)
    self.game_counter += 1

    # Decay epsilon
    self.epsilon = max(FINAL_EPSILON, self.epsilon * EPSILON_DECAY)

    # Print epsilon and average score every 100 games
    if self.game_counter % 100 == 0:
        print(f"Game {self.game_counter}: Current epsilon value: {self.epsilon:.4f}")
        avg_score = np.mean(self.scores[-100:])
        self.average_scores.append(avg_score)
        print(f"Average score over the last 100 games: {avg_score}")
        create_performance_graph(self)

def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate the reward based on the events that occurred during the game.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        MOVED_TOWARDS_COIN: 2,
        MOVED_AWAY_FROM_COIN: -2,
        EXPLORED_NEW_POSITION: 3,
        e.INVALID_ACTION: -5,
        e.WAITED: -1,
        e.KILLED_SELF: -1000
    }
    reward_sum = 0
    for event in events:
        reward_sum += game_rewards.get(event, 0)
    return reward_sum

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