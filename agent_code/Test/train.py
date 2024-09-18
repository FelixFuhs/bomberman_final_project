import os
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import events as e
from .callbacks import state_to_features, ACTIONS, create_graphs, count_crates_destroyed  # Import count_crates_destroyed
from .config import (TRANSITION_HISTORY_SIZE, BATCH_SIZE, GAMMA, TARGET_UPDATE, 
                     LEARNING_RATE, TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_DECAY)

# Transition tuple for storing experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    """Initialize training settings."""
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.temperature = TEMPERATURE_START  # Initialize temperature
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.MSELoss()
    self.total_training_steps = 0
    self.losses = []
    self.scores = []
    self.game_counter = 0
    self.logger.info(f"Training initialized. Temperature starts at {self.temperature}, learning rate at {LEARNING_RATE}")

    # Initialize tracking metrics
    self.total_rewards = []
    self.positions_visited = set()  # To track unique positions visited in an episode 
    self.coordinate_history = deque([], 15)  # Track the last 15 positions
    self.rewards_episode = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Process game events and store transitions."""
    self.logger.debug(f"Events: {events} at step {new_game_state['step']}")

    # Convert game states to features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events, new_game_state)

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
    if self.total_training_steps % TARGET_UPDATE == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.info(f"Target network updated at step {self.total_training_steps}")

    # Update temperature for exploration-exploitation balance
    self.temperature = max(TEMPERATURE_END, TEMPERATURE_START * np.exp(-1. * self.total_training_steps / TEMPERATURE_DECAY))
    self.logger.debug(f"Temperature decayed to {self.temperature:.4f}")
    self.total_training_steps += 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Handle end of round logic."""
    self.logger.info(f"End of round. Events: {events}")

    # Final transition
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events, last_game_state)
    if last_features is not None:
        self.transitions.append(Transition(last_features, last_action, None, reward))
        self.logger.debug(f"Final transition stored. Buffer size: {len(self.transitions)}")

    # Perform final optimization step
    if len(self.transitions) >= BATCH_SIZE:
        loss = optimize_model(self)
        if loss:
            self.losses.append(loss)

    # Save the model with exception handling
    try:
        torch.save(self.q_network.state_dict(), "dqn_model.pt")
        self.logger.info(f"Model saved after round {self.game_counter}")
    except Exception as e:
        self.logger.error(f"Error saving model: {e}")

    # Track game performance
    self.scores.append(last_game_state['self'][1])

    # Track total rewards
    total_reward = sum(self.rewards_episode)
    self.total_rewards.append(total_reward)

    # Reset rewards and positions for the next episode
    self.rewards_episode = []
    self.positions_visited = set()
    self.coordinate_history = deque([], 15)

    # Create graphs every 100 games
    if self.game_counter % 100 == 0:
        avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
        avg_reward = np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards)
        self.logger.info(f"Game {self.game_counter}: Temperature {self.temperature:.4f}, Avg. score (last 100 games): {avg_score:.2f}, Avg. reward: {avg_reward:.2f}")
        create_graphs(self)  # Use create_graphs(self) instead of self.create_graphs()

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

    # Convert the list of non-final next states to a tensor
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], device=self.device, dtype=torch.float32)

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

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

def reward_from_events(self, events: List[str], game_state: dict) -> float:
    """Translate game events into rewards."""
    game_rewards = {
        e.COIN_COLLECTED: 7,        # Increased reward to encourage collecting coins
        e.KILLED_OPPONENT: 15,      # Increased reward for killing opponents
        e.INVALID_ACTION: -2,       # Increased penalty for invalid actions
        e.WAITED: -1,               # Increased penalty for waiting
        e.KILLED_SELF: -10,         # Increased penalty for self-destruction
        e.SURVIVED_ROUND: 1,        # Increased reward for survival
        e.CRATE_DESTROYED: 3,       # Reward per crate destroyed
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.MOVED_UP: -0.05,
        e.BOMB_DROPPED: 0,          # Base reward for dropping bombs, adjusted below
        e.GOT_KILLED: -5,           # Penalty for being killed
        e.OPPONENT_ELIMINATED: 5,   # Reward for eliminating an opponent
        'MOVED_TO_NEW_POSITION': 0.2,       # Small reward for exploring
        'MOVED_TO_RECENT_POSITION': -2      # Penalty for revisiting recent positions
    }

    # Check for custom events
    position = game_state['self'][3]
    x, y = position
    arena = game_state['field']

    if position not in self.positions_visited:
        events.append('MOVED_TO_NEW_POSITION')
    else:
        if position in self.coordinate_history:
            events.append('MOVED_TO_RECENT_POSITION')

    # Update position tracking
    self.positions_visited.add(position)
    self.coordinate_history.append(position)

    # Adjust reward for BOMB_DROPPED
    if e.BOMB_DROPPED in events:
        crates_destroyed = count_crates_destroyed(arena, x, y)
        if crates_destroyed > 0:
            # Positive reward proportional to potential crates destroyed
            bomb_reward = crates_destroyed * 2  # Adjust coefficient as needed
            game_rewards[e.BOMB_DROPPED] = bomb_reward
        else:
            # Negative reward for unnecessary bomb
            game_rewards[e.BOMB_DROPPED] = -2

    # Additional penalties to prevent self-destructive behavior
    if 'MOVED_TO_RECENT_POSITION' in events:
        game_rewards['MOVED_TO_RECENT_POSITION'] = -2

    # Custom Reward: Punish moving inside bomb blast radius
    # Base punishment
    base_punishment = -0.5

    # Bomb blast radius
    blast_radius = 3

    # Maximum bomb timer to consider for scaling
    max_time = 3

    # Initialize punishment
    punishment = 0.0

    # Agent's position
    agent_x, agent_y = x, y

    for bomb in game_state['bombs']:
        (bx, by), t = bomb
        distance = abs(agent_x - bx) + abs(agent_y - by)  # Manhattan distance
        if distance <= blast_radius and t <= max_time:
            # Scale punishment by closeness and urgency
            distance_scale = (blast_radius - distance + 1) / (blast_radius + 1)  # Closer distance -> higher punishment
            time_scale = (max_time - t + 1) / (max_time + 1)  # Less time remaining -> higher punishment
            punishment += base_punishment * distance_scale * time_scale

    # Add the punishment to the total reward
    reward = sum(game_rewards.get(event, 0) for event in events) + punishment

    # Track rewards for logging
    self.rewards_episode.append(reward)

    return reward
