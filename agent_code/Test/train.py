# train.py

import os
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import events as e
from .callbacks import (
    state_to_features, ACTIONS, create_graphs, count_crates_destroyed,
    get_escape_routes  # Import get_escape_routes
)
from .config import (
    TRANSITION_HISTORY_SIZE, BATCH_SIZE, GAMMA, TARGET_UPDATE,
    LEARNING_RATE, TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_DECAY
)
import settings as s  # Import settings

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

    # Initialize distance tracking for coins and enemies
    self.previous_coin_distance = None
    self.previous_opponent_distance = None

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Process game events and store transitions."""
    self.logger.debug(f"Events: {events} at step {new_game_state['step']}")

    # Convert game states to features
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    # Detect 'ESCAPED_BOMB' event
    if old_game_state is not None and new_game_state is not None:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]
        bomb_map = create_bomb_map(old_game_state)
        if bomb_map[old_pos] <= 3 and bomb_map[new_pos] > bomb_map[old_pos]:
            events.append('ESCAPED_BOMB')

    reward = self.reward_from_events(events, new_game_state)

    # Store transition in replay buffer
    if old_features is not None and new_features is not None:
        self.transitions.append(Transition(old_features, self_action, new_features, reward))
        self.logger.debug(f"Stored transition. Buffer size: {len(self.transitions)}")

    # Optimize the model if we have enough samples
    if len(self.transitions) >= BATCH_SIZE:
        loss = self.optimize_model()
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
    last_features = state_to_features(self, last_game_state)
    reward = self.reward_from_events(events, last_game_state)
    if last_features is not None:
        self.transitions.append(Transition(last_features, last_action, None, reward))
        self.logger.debug(f"Final transition stored. Buffer size: {len(self.transitions)}")

    # Perform final optimization step
    if len(self.transitions) >= BATCH_SIZE:
        loss = self.optimize_model()
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
    self.previous_coin_distance = None
    self.previous_opponent_distance = None

    # Create graphs every 100 games
    if self.game_counter % 100 == 0:
        avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
        avg_reward = np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards)
        self.logger.info(f"Game {self.game_counter}: Temperature {self.temperature:.4f}, Avg. score (last 100 games): {avg_score:.2f}, Avg. reward: {avg_reward:.2f}")
        create_graphs(self)

    self.game_counter += 1

def optimize_model(self):
    """Perform a single optimization step."""
    if len(self.transitions) < BATCH_SIZE:
        return None

    # Sample a batch from memory
    transitions = random.sample(self.transitions, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare batches
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], device=self.device, dtype=torch.float32)
    state_batch = torch.tensor(np.array(batch.state), device=self.device, dtype=torch.float32)
    action_batch = torch.tensor([ACTIONS.index(a) for a in batch.action], device=self.device, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

    # Compute Q(s, a)
    state_action_values = self.q_network(state_batch).gather(1, action_batch)

    # Compute expected Q values
    next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
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
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 20,
        e.INVALID_ACTION: -2,
        e.WAITED: -1,
        e.KILLED_SELF: -30,
        e.SURVIVED_ROUND: 5,
        e.CRATE_DESTROYED: 8,
        e.BOMB_DROPPED: 0,  # Base reward, adjusted below
        e.GOT_KILLED: -20,
        e.OPPONENT_ELIMINATED: 10,
        'MOVED_TO_NEW_POSITION': 0.5,
        'MOVED_TO_RECENT_POSITION': -5,
        'ESCAPED_BOMB': 10,
        'STAYED_IN_DANGER': -5,
        'UNSAFE_BOMB_PLACEMENT': -10,
        'REDUCED_DISTANCE_TO_COIN': 1,
        'INCREASED_DISTANCE_TO_COIN': -1,
        'REDUCED_DISTANCE_TO_ENEMY': 1,
        'INCREASED_DISTANCE_TO_ENEMY': -1,
    }

    # Position and state information
    position = game_state['self'][3]
    x, y = position
    arena = game_state['field']
    coins = game_state['coins']
    others = [xy for (_, _, _, xy) in game_state['others']]

    # Loop detection
    if position not in self.positions_visited:
        events.append('MOVED_TO_NEW_POSITION')
    else:
        if position in self.coordinate_history:
            events.append('MOVED_TO_RECENT_POSITION')

    # Update position tracking
    self.positions_visited.add(position)
    self.coordinate_history.append(position)

    # Distance to nearest coin
    if coins:
        distances = [abs(cx - x) + abs(cy - y) for (cx, cy) in coins]
        nearest_coin_distance = min(distances)
        if self.previous_coin_distance is not None:
            if nearest_coin_distance < self.previous_coin_distance:
                events.append('REDUCED_DISTANCE_TO_COIN')
            elif nearest_coin_distance > self.previous_coin_distance:
                events.append('INCREASED_DISTANCE_TO_COIN')
        self.previous_coin_distance = nearest_coin_distance
    else:
        self.previous_coin_distance = None

    # Distance to nearest opponent
    if others:
        distances = [abs(ox - x) + abs(oy - y) for (ox, oy) in others]
        nearest_opponent_distance = min(distances)
        if self.previous_opponent_distance is not None:
            if nearest_opponent_distance < self.previous_opponent_distance:
                events.append('REDUCED_DISTANCE_TO_ENEMY')
            elif nearest_opponent_distance > self.previous_opponent_distance:
                events.append('INCREASED_DISTANCE_TO_ENEMY')
        self.previous_opponent_distance = nearest_opponent_distance
    else:
        self.previous_opponent_distance = None

    # Adjust reward for BOMB_DROPPED
    if e.BOMB_DROPPED in events:
        # Check if bomb placement is safe
        escape_routes_after_bomb = get_escape_routes(arena, x, y, create_bomb_map(game_state))
        if escape_routes_after_bomb == 0:
            events.append('UNSAFE_BOMB_PLACEMENT')
        else:
            crates_destroyed = count_crates_destroyed(arena, x, y)
            if crates_destroyed > 0:
                # Positive reward proportional to potential crates destroyed
                bomb_reward = crates_destroyed * 3
                game_rewards[e.BOMB_DROPPED] = bomb_reward
            else:
                # Negative reward for unnecessary bomb
                game_rewards[e.BOMB_DROPPED] = -5

    # Penalty for staying in danger
    bomb_map = create_bomb_map(game_state)
    if bomb_map[x, y] <= 3:
        events.append('STAYED_IN_DANGER')

    # Compute the total reward
    reward = sum(game_rewards.get(event, 0) for event in events)

    # Additional punishment for being close to bombs
    punishment = 0.0
    for bomb in game_state['bombs']:
        (bx, by), t = bomb
        distance = abs(x - bx) + abs(y - by)
        if distance <= s.BOMB_POWER and t <= s.BOMB_TIMER - 1:
            distance_scale = (s.BOMB_POWER - distance + 1) / (s.BOMB_POWER + 1)
            time_scale = (s.BOMB_TIMER - t) / s.BOMB_TIMER
            punishment -= 0.5 * distance_scale * time_scale

    reward += punishment

    # Track rewards for logging
    self.rewards_episode.append(reward)

    return reward

def create_bomb_map(game_state):
    """Create a bomb map similar to the one in state_to_features."""
    arena = game_state['field']
    bombs = game_state['bombs']
    bomb_map = np.ones(arena.shape) * (s.BOMB_TIMER + 1)
    for (bx, by), t in bombs:
        for (i, j) in [(bx + h, by) for h in range(-s.BOMB_POWER, s.BOMB_POWER + 1)] + [(bx, by + h) for h in range(-s.BOMB_POWER, s.BOMB_POWER + 1)]:
            if 0 <= i < arena.shape[0] and 0 <= j < arena.shape[1]:
                bomb_map[i, j] = min(bomb_map[i, j], t)
    return bomb_map
