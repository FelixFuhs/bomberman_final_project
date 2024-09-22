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
    state_to_features, ACTIONS, create_graphs,
    count_crates_destroyed, get_escape_routes, get_bomb_map
)
from .config import (
    TRANSITION_HISTORY_SIZE, BATCH_SIZE, GAMMA, TARGET_UPDATE,
    LEARNING_RATE, TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_DECAY
)

# Transition tuple for storing experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    """Initialize training settings."""
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.temperature = TEMPERATURE_START  # Initialize temperature
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.SmoothL1Loss().to(self.device)  # Move criterion to device
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
    self.last_game_state = None  # For tracking previous state in game_events_occurred
    self.last_action = None  # For tracking previous action in game_events_occurred

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Process game events and store transitions."""
    self.logger.debug(f"Events: {events} at step {new_game_state['step']}")

    # Convert game states to features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events, new_game_state, old_game_state)

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
    self.temperature = max(
        TEMPERATURE_END,
        TEMPERATURE_START * np.exp(-1. * self.total_training_steps / TEMPERATURE_DECAY)
    )
    self.logger.debug(f"Temperature decayed to {self.temperature:.4f}")
    self.total_training_steps += 1

    # Update last_game_state and last_action for next call
    self.last_game_state = old_game_state
    self.last_action = self_action

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to finalize training."""
    self.logger.info(f"End of round. Events: {events}")

    # Check if the agent won the game
    agent_won = 'GAME_WON' in events

    # Final transition
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events, last_game_state, self.last_game_state)
    if last_features is not None:
        self.transitions.append(Transition(last_features, last_action, None, reward))
        self.logger.debug(f"Final transition stored with reward {reward}. Buffer size: {len(self.transitions)}")

    # Perform final optimization step
    if len(self.transitions) >= BATCH_SIZE:
        loss = optimize_model(self)
        if loss:
            self.losses.append(loss)

    # Save the model with exception handling
    try:
        torch.save(self.q_network.state_dict(), "dqn_model.pt")
        self.logger.info(f"Model saved after round {self.game_counter}")
    except Exception as exc:
        self.logger.error(f"Error saving model: {exc}")

    # Track game performance
    self.scores.append(last_game_state['self'][1])

    # Track total rewards
    total_reward = sum(self.rewards_episode)
    self.total_rewards.append(total_reward)

    # Reset rewards and positions for the next episode
    self.rewards_episode = []
    self.positions_visited = set()
    self.coordinate_history = deque([], 15)
    self.last_game_state = None
    self.last_action = None

    # Create graphs every 100 games
    if self.game_counter % 100 == 0:
        avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
        avg_reward = np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards)
        self.logger.info(f"Game {self.game_counter}: Temperature {self.temperature:.4f}, Avg. score (last 100 games): {avg_score:.2f}, Avg. reward: {avg_reward:.2f}")
        self.create_graphs()

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
    non_final_next_states = torch.tensor(
        [s for s in batch.next_state if s is not None],
        device=self.device,
        dtype=torch.float32
    )

    state_batch = torch.tensor(np.array(batch.state), device=self.device, dtype=torch.float32)
    action_batch = torch.tensor(
        [ACTIONS.index(a) for a in batch.action],
        device=self.device,
        dtype=torch.long
    ).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

    # Compute Q(s, a) using the main network
    state_action_values = self.q_network(state_batch).gather(1, action_batch)

    # Compute Q(s', a') for non-final states using Double DQN
    with torch.no_grad():
        next_state_actions = self.q_network(non_final_next_states).max(1)[1].unsqueeze(1)
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).gather(1, next_state_actions).squeeze()

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss using Huber loss
    loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

def reward_from_events(self, events: List[str], game_state: dict, old_game_state: dict) -> float:
    """Translate game events into scaled rewards."""
    self.logger.debug(f"Events received for reward calculation: {events}")

    game_rewards = {
        e.COIN_COLLECTED: 30,        # Increased reward to encourage collecting coins
        e.KILLED_OPPONENT: 25,       # Increased reward for killing opponents
        e.INVALID_ACTION: -6,        # Increased penalty for invalid actions
        e.WAITED: -1,                # Penalty for waiting
        e.KILLED_SELF: -20,          # Increased penalty for self-destruction
        e.SURVIVED_ROUND: 2,         # Reward for survival
        e.CRATE_DESTROYED: 2,        # Reward per crate destroyed
        e.BOMB_DROPPED: 0,           # Base reward for dropping bombs, adjusted below
        e.GOT_KILLED: -20,           # Penalty for being killed
        e.OPPONENT_ELIMINATED: 5,    # Reward for eliminating an opponent
        'MOVED_TO_NEW_POSITION': 0.5,      # Small reward for exploring
        'MOVED_TO_RECENT_POSITION': -2,    # Penalty for revisiting recent positions
        'GAME_WON': 500,                   # Huge reward for winning the game
        'MOVED_TOWARDS_COIN': 1,           # Reward for moving towards coin
        'MOVED_AWAY_FROM_COIN': -1,        # Penalty for moving away from coin
        'SAFE_BOMB_DROP': 5,               # Reward for safe bomb drop
        'DANGEROUS_BOMB_DROP': -5,         # Penalty for dangerous bomb drop
        'STAYED_IN_BLAST_ZONE': -5,        # Penalty for staying in blast zone
        'ESCAPED_BLAST_ZONE': 5,           # Reward for escaping blast zone
        'TIME_PENALTY': -0.1               # Small penalty for each step
    }

    # Implement time-based penalty
    events.append('TIME_PENALTY')

    position = game_state['self'][3]
    x, y = position
    arena = game_state['field']

    # Check for custom events
    if old_game_state is not None:
        # Determine if agent moved towards or away from nearest coin
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(game_state)

        # Distance to nearest coin before and after
        old_coin_dx = old_features[5]
        old_coin_dy = old_features[6]
        new_coin_dx = new_features[5]
        new_coin_dy = new_features[6]

        old_coin_distance = abs(old_coin_dx) + abs(old_coin_dy)
        new_coin_distance = abs(new_coin_dx) + abs(new_coin_dy)

        if new_coin_distance < old_coin_distance:
            events.append('MOVED_TOWARDS_COIN')
        elif new_coin_distance > old_coin_distance:
            events.append('MOVED_AWAY_FROM_COIN')

    # Determine if the move was to a new or recent position
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
        is_safe = is_bomb_placement_safe(game_state)
        if crates_destroyed > 0 and is_safe:
            # Positive reward for safe bomb drop
            events.append('SAFE_BOMB_DROP')
        else:
            # Negative reward for dangerous bomb drop
            events.append('DANGEROUS_BOMB_DROP')

    # Check if agent stayed in or escaped blast zone
    if old_game_state is not None:
        old_bomb_map = get_bomb_map(old_game_state)
        new_bomb_map = get_bomb_map(game_state)
        old_in_blast_zone = old_bomb_map[x, y] <= 3
        new_in_blast_zone = new_bomb_map[x, y] <= 3
        if old_in_blast_zone and not new_in_blast_zone:
            events.append('ESCAPED_BLAST_ZONE')
        elif new_in_blast_zone:
            events.append('STAYED_IN_BLAST_ZONE')

    # Compute total reward
    reward = sum(game_rewards.get(event, 0) for event in events)

    # Track rewards for logging
    self.rewards_episode.append(reward)
    self.logger.debug(f"Calculated reward: {reward}")

    return reward

def is_bomb_placement_safe(game_state):
    """Check if placing a bomb at the agent's position is safe."""
    x, y = game_state['self'][3]
    bomb_map = get_bomb_map(game_state)
    escape_routes = get_escape_routes(game_state['field'], x, y, bomb_map, game_state)
    return escape_routes > 0

def get_bomb_map(game_state):
    """Utility function to get bomb map for the game state."""
    arena = game_state['field']
    bombs = game_state['bombs']
    bomb_map = np.ones(arena.shape) * 5  # Distance to the nearest bomb

    # Update bomb map with bombs that will explode
    for (bx, by), t in bombs:
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            for i in range(1, 4):
                nx, ny = bx + dx*i, by + dy*i
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    if arena[nx, ny] == -1:
                        break
                    bomb_map[nx, ny] = min(bomb_map[nx, ny], t)
                    if arena[nx, ny] != 0:
                        break
                else:
                    break

    # Update bomb map with current explosions
    explosion_map = game_state['explosion_map']
    bomb_map[explosion_map > 0] = 0  # Explosion is currently at these tiles
    return bomb_map