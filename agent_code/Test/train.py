import os
import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS, DQN, create_performance_graph, create_loss_graph

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1_000_000  # Increased to store more experiences
BATCH_SIZE = 512  # Increased for more stable gradients
GAMMA = 0.99  # This is good, no change needed
TARGET_UPDATE = 1000  # Increased for more stable target network
LEARNING_RATE = 0.0001  # Reduced for more stable learning
EPSILON_START = 1.0  # This is good
EPSILON_END = 0.01  # Slightly lower for more exploitation in late stages
EPSILON_DECAY = 50000  # Increased for slower decay

# Custom events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
BOMB_PLACED_NEAR_CRATES = "BOMB_PLACED_NEAR_CRATES"
ESCAPED_FROM_BOMB = "ESCAPED_FROM_BOMB"
MOVED_TOWARDS_OPPONENT = "MOVED_TOWARDS_OPPONENT"
MOVED_AWAY_FROM_OPPONENT = "MOVED_AWAY_FROM_OPPONENT"
INVALID_MOVE_ATTEMPT = "INVALID_MOVE_ATTEMPT"  # New custom event

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.epsilon = EPSILON_START
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.MSELoss()
    self.losses = []  # Initialize the losses list

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Add custom events
    events.extend(get_custom_events(old_game_state, self_action, new_game_state))

    # State features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # Compute reward
    reward = reward_from_events(self, events)

    # Store the transition in memory
    if old_features is not None and new_features is not None:
        self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Perform one step of the optimization
    if len(self.transitions) > BATCH_SIZE:
        loss = optimize_model(self)
        if loss is not None:
            self.losses.append(loss)  # Record the loss

    # Update the target network, copying all weights and biases in DQN
    if self.steps % TARGET_UPDATE == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Update epsilon
    self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
        np.exp(-1. * self.steps / EPSILON_DECAY)
    self.steps += 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Add custom events for the final step
    events.extend(get_custom_events(last_game_state, last_action, None))

    # Compute final reward
    reward = reward_from_events(self, events)

    # Store the final transition
    last_features = state_to_features(last_game_state)
    if last_features is not None:
        self.transitions.append(Transition(last_features, last_action, None, reward))

    # Perform final optimization step
    if len(self.transitions) > BATCH_SIZE:
        loss = optimize_model(self)
        if loss is not None:
            self.losses.append(loss)

    # Store the model
    torch.save(self.q_network.state_dict(), "dqn_model.pt")

    # Reset steps_since_bomb for the next round
    self.steps_since_bomb = 0

    # Update performance tracking
    self.game_counter += 1
    score = last_game_state['self'][1]
    self.scores.append(score)

    # Print epsilon and average score every 100 games
    if self.game_counter % 100 == 0:
        print(f"Game {self.game_counter}: Current epsilon value: {self.epsilon:.4f}")
        avg_score = np.mean(self.scores[-100:])
        self.average_scores.append(avg_score)
        print(f"Average score over the last 100 games: {avg_score:.2f}")
        create_performance_graph(self)
        
        # Create the loss graph every 100 games
        if len(self.losses) > 0:
            create_loss_graph(self)
            self.losses = []  # Reset losses after creating the graph

    # Reset epsilon for the next round
    self.epsilon = EPSILON_START

def optimize_model(self):
    if len(self.transitions) < BATCH_SIZE:
        return None

    transitions = random.sample(self.transitions, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.tensor(np.array([s for s in batch.next_state if s is not None]), device=self.device, dtype=torch.float32)

    state_batch = torch.tensor(np.array(batch.state), device=self.device, dtype=torch.float32)
    action_batch = torch.tensor([ACTIONS.index(action) for action in batch.action], device=self.device, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

    # Predicted Q-values from the Q-network for the chosen actions
    state_action_values = self.q_network(state_batch).gather(1, action_batch)

    # Target Q-values using the target network and next state
    next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
    next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute the loss (MSE loss between predicted Q-values and target Q-values)
    loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagation and optimization
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.q_network.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    return loss.item()  # Return the loss value


def get_custom_events(old_game_state, self_action, new_game_state):
    custom_events = []

    if old_game_state and new_game_state:
        # Check for movement towards/away from coin
        old_coin_distance = get_nearest_coin_distance(old_game_state)
        new_coin_distance = get_nearest_coin_distance(new_game_state)
        if new_coin_distance < old_coin_distance:
            custom_events.append(MOVED_TOWARDS_COIN)
        elif new_coin_distance > old_coin_distance:
            custom_events.append(MOVED_AWAY_FROM_COIN)

        # Check for bomb placement near crates
        if self_action == 'BOMB':
            crates_nearby = count_nearby_crates(new_game_state)
            if crates_nearby > 1:
                custom_events.append(f"BOMB_PLACED_NEAR_{crates_nearby}_CRATES")

        # Check for escaping from bomb
        if in_danger(old_game_state) and not in_danger(new_game_state):
            custom_events.append(ESCAPED_FROM_BOMB)

        # Check for movement relative to opponent
        old_opponent_distance = get_nearest_opponent_distance(old_game_state)
        new_opponent_distance = get_nearest_opponent_distance(new_game_state)
        distance_change = old_opponent_distance - new_opponent_distance
        custom_events.append(f"OPPONENT_DISTANCE_CHANGE_{distance_change:.2f}")

        # Check for invalid move attempts
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]
        if old_pos == new_pos and self_action in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            custom_events.append(INVALID_MOVE_ATTEMPT)

    return custom_events

def reward_from_events(self, events: List[str]) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        MOVED_TOWARDS_COIN: 0.05,
        MOVED_AWAY_FROM_COIN: -0.05,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.2,
        e.KILLED_SELF: -5,
        ESCAPED_FROM_BOMB: 0.7,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.2,
        e.SURVIVED_ROUND: 2,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.MOVED_DOWN: -0.05,
        e.MOVED_UP: -0.05,
        INVALID_MOVE_ATTEMPT: -0.2
    }
    
    reward = 0
    for event in events:
        if event.startswith("BOMB_PLACED_NEAR_"):
            crates = int(event.split("_")[-2])
            reward += min(0.1 * crates, 0.5)  # Cap at 0.5
        elif event.startswith("OPPONENT_DISTANCE_CHANGE_"):
            change = float(event.split("_")[-1])
            reward += change * 0.1  # Reward for getting closer, penalize for moving away
        else:
            reward += game_rewards.get(event, 0)
    
    return np.clip(reward, -1, 1)  # Clip the final reward to [-1, 1]

# Helper functions
def get_nearest_coin_distance(game_state):
    agent_pos = game_state['self'][3]
    coins = game_state['coins']
    if not coins:
        return float('inf')
    return min(manhattan_distance(agent_pos, coin) for coin in coins)

def count_nearby_crates(game_state):
    agent_pos = game_state['self'][3]
    arena = game_state['field']
    return sum(arena[agent_pos[0] + dx, agent_pos[1] + dy] == 1
               for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)])

def in_danger(game_state):
    agent_pos = game_state['self'][3]
    bombs = game_state['bombs']
    for bomb_pos, _ in bombs:
        if manhattan_distance(agent_pos, bomb_pos) <= 3:
            return True
    return False

def get_nearest_opponent_distance(game_state):
    agent_pos = game_state['self'][3]
    others = game_state['others']
    if not others:
        return float('inf')
    return min(manhattan_distance(agent_pos, other[3]) for other in others)

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])