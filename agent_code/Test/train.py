import os
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import events as e
from .callbacks import state_to_features, ACTIONS, create_graphs

# Transition tuple for storing experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 100_000  # Max transitions stored in replay buffer
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10
LEARNING_RATE = 0.0005
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000

def setup_training(self):
    """Initialize training settings."""
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.epsilon = EPSILON_START
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.MSELoss()
    self.steps = 0
    self.losses = []
    self.scores = []
    self.game_counter = 0
    self.logger.info(f"Training initialized. Epsilon starts at {self.epsilon}, learning rate at {LEARNING_RATE}")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Process game events and store transitions."""
    self.logger.debug(f"Events: {events} at step {new_game_state['step']}")
    
    # Convert game states to features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    # Store transition in replay buffer
    if old_features is not None and new_features is not None:
        self.transitions.append(Transition(old_features, self_action, new_features, reward))
        self.logger.debug(f"Stored transition. Buffer size: {len(self.transitions)}")

    # Optimize the model if we have enough samples
    if len(self.transitions) >= BATCH_SIZE:
        loss = optimize_model(self)
        if loss is not None:
            self.losses.append(loss)
            self.logger.debug(f"Optimization done. Loss: {loss}")

    # Update target network every few steps
    if self.steps % TARGET_UPDATE == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.info(f"Target network updated at step {self.steps}")

    # Update epsilon for exploration-exploitation balance
    self.epsilon = max(EPSILON_END, EPSILON_START * np.exp(-1. * self.steps / EPSILON_DECAY))
    self.logger.debug(f"Epsilon decayed to {self.epsilon:.4f}")
    self.steps += 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Handle end of round logic."""
    self.logger.info(f"End of round. Events: {events}")

    # Final transition
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    if last_features is not None:
        self.transitions.append(Transition(last_features, last_action, None, reward))
        self.logger.debug(f"Final transition stored. Buffer size: {len(self.transitions)}")

    # Perform final optimization step
    if len(self.transitions) >= BATCH_SIZE:
        loss = optimize_model(self)
        if loss:
            self.losses.append(loss)

    # Save the model
    torch.save(self.q_network.state_dict(), "dqn_model.pt")
    self.logger.info(f"Model saved after round {self.game_counter}")

    # Track game performance and update graphs periodically
    self.scores.append(last_game_state['self'][1])

    # Only create graphs every 100 games (or adjust this to a reasonable interval)
    if self.game_counter % 100 == 0:
        avg_score = np.mean(self.scores[-100:])
        self.logger.info(f"Game {self.game_counter}: Epsilon {self.epsilon:.4f}, Avg. score (last 100 games): {avg_score:.2f}")
        create_graphs(self)
        self.losses = []  # Reset losses after creating the graph

    self.game_counter += 1


def optimize_model(self):
    """Perform a single optimization step."""
    if len(self.transitions) < BATCH_SIZE:
        return None

    # Sample a batch from memory
    transitions = random.sample(self.transitions, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Mask for non-final states
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)

    # Convert the list of non-final next states to a numpy array before converting to a tensor
    non_final_next_states = np.array([s for s in batch.next_state if s is not None])
    non_final_next_states = torch.tensor(non_final_next_states, device=self.device, dtype=torch.float32)
    
    state_batch = torch.tensor(np.array(batch.state), device=self.device, dtype=torch.float32)
    action_batch = torch.tensor([ACTIONS.index(a) for a in batch.action], device=self.device, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

    # Compute Q(s, a)
    state_action_values = self.q_network(state_batch).gather(1, action_batch)

    # Compute Q(s', a') for non-final states
    next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagate the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()


def reward_from_events(self, events: List[str]) -> int:
    """Translate game events into rewards."""
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -10,
        e.WAITED: -5,
        e.KILLED_SELF: -100,
        e.SURVIVED_ROUND: 50,
        e.CRATE_DESTROYED: 10,
    }
    return sum(game_rewards.get(event, 0) for event in events)
