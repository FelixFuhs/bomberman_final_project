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
from .callbacks import state_to_features, ACTIONS, create_graphs, count_crates_destroyed, is_move_valid
from .config import (TRANSITION_HISTORY_SIZE, BATCH_SIZE, GAMMA, TARGET_UPDATE,
                     LEARNING_RATE, TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_DECAY)

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

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Process game events and store transitions."""
    self.logger.debug(f"Events: {events} at step {new_game_state['step']}")

    # Append TIME_PENALTY
    events.append('TIME_PENALTY')

    # Convert game states to features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if old_features is not None and new_features is not None:
        # Extract positions
        old_position = old_game_state['self'][3]
        new_position = new_game_state['self'][3]

        # Compute distances to nearest coin
        old_coin_distance = compute_distance_to_nearest_coin(old_game_state)
        new_coin_distance = compute_distance_to_nearest_coin(new_game_state)

        if old_coin_distance is not None and new_coin_distance is not None:
            if new_coin_distance < old_coin_distance:
                events.append('MOVED_TOWARDS_COIN')
            elif new_coin_distance > old_coin_distance:
                events.append('MOVED_AWAY_FROM_COIN')

        # Determine if the agent is in a blast zone
        old_in_blast_zone = is_in_blast_zone(old_game_state)
        new_in_blast_zone = is_in_blast_zone(new_game_state)

        # Check transitions
        if not old_in_blast_zone and new_in_blast_zone:
            events.append('MOVED_INTO_BLAST_ZONE')
        elif old_in_blast_zone and new_in_blast_zone:
            events.append('STAYED_IN_BLAST_ZONE')
        elif old_in_blast_zone and not new_in_blast_zone:
            events.append('ESCAPED_BLAST_ZONE')

        # Store transition in replay buffer
        reward = reward_from_events(self, events, new_game_state)
        self.transitions.append(Transition(old_features, self_action, new_features, reward))
        self.logger.debug(f"Stored transition. Buffer size: {len(self.transitions)}")
    else:
        reward = reward_from_events(self, events, new_game_state)

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
    self.logger.info(f"End of round. Events: {events}")

    # Initialize variables
    agent_won = False

    # Check if the agent survived
    agent_survived = e.KILLED_SELF not in events and e.GOT_KILLED not in events
    self.logger.info(f"Agent survived: {agent_survived}")

    # Extract agent's name and score
    agent_name = last_game_state['self'][0]
    agent_score = last_game_state['self'][1]
    agent_step = last_game_state['step']  # Assuming step count is available

    self.logger.info(f"Agent '{agent_name}' score: {agent_score}")

    other_agents = last_game_state.get('others', [])
    self.logger.info(f"Other agents: {other_agents}")

    if len(other_agents) == 0:
        # Single-player mode
        if agent_survived and len(last_game_state.get('coins', [])) == 0:
            # Agent survived and collected all the coins
            agent_won = True
            self.logger.info(f"Agent '{agent_name}' has won the game by collecting all coins.")
        else:
            self.logger.info(f"Agent '{agent_name}' did not win the game in single-player mode.")
    else:
        # Multi-player mode
        # Collect scores and steps of all agents, along with survival status
        all_agents = [(agent_name, agent_score, agent_step, agent_survived)]  # Include our agent

        for other_agent in other_agents:
            other_agent_name = other_agent[0]
            other_agent_score = other_agent[1]
            other_agent_step = last_game_state.get('agent_steps', {}).get(other_agent_name, last_game_state['step'])
            other_agent_alive = other_agent[3] != (-1, -1)  # Assuming dead agents have position (-1, -1)
            all_agents.append((other_agent_name, other_agent_score, other_agent_step, other_agent_alive))
            self.logger.info(f"Other agent '{other_agent_name}' score: {other_agent_score}, step reached: {other_agent_step}, alive: {other_agent_alive}")

        # Filter out agents who are dead
        surviving_agents = [agent for agent in all_agents if agent[3]]  # Only include alive agents

        if not surviving_agents:
            # No agents survived, no winner
            self.logger.info("No agents survived. No winner.")
        else:
            # Determine the agent with the highest score among surviving agents
            max_score = max(score for _, score, _, _ in surviving_agents)
            agents_with_max_score = [agent for agent in surviving_agents if agent[1] == max_score]

            self.logger.info(f"Surviving agents with max score {max_score}: {agents_with_max_score}")

            if len(agents_with_max_score) == 1:
                # Only one agent has the highest score among surviving agents
                winner = agents_with_max_score[0]
                if winner[0] == agent_name:
                    agent_won = True
                    self.logger.info(f"Agent '{agent_name}' has won the game with the highest score among surviving agents.")
                else:
                    self.logger.info(f"Agent '{agent_name}' did not win. Winner is '{winner[0]}'.")
            else:
                # Tie situation among surviving agents: Agent who reached the score first wins
                earliest_step = min(step for _, _, step, _ in agents_with_max_score)
                winners = [agent for agent in agents_with_max_score if agent[2] == earliest_step]
                winner = winners[0]  # Assuming there's only one agent who reached the earliest step
                self.logger.info(f"Tie detected among surviving agents. Agent(s) who reached the score first: {winners}")

                if winner[0] == agent_name:
                    agent_won = True
                    self.logger.info(f"Agent '{agent_name}' has won the game by reaching the highest score first among surviving agents.")
                else:
                    self.logger.info(f"Agent '{agent_name}' did not win. Winner is '{winner[0]}'.")

    if agent_won:
        events.append('GAME_WON')
        self.logger.info("Agent has won the game!")

    # Final transition
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events, last_game_state)
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
    self.scores.append(agent_score)

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
        create_graphs(self)

    self.game_counter += 1

    # Reset agent score steps for the next round
    self.agent_score_steps = {}

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

def reward_from_events(self, events: List[str], game_state: dict) -> float:
    """Translate game events into scaled rewards."""
    self.logger.debug(f"Events received for reward calculation: {events}")

    game_rewards = {
        e.COIN_COLLECTED: 30,        # Increased reward to encourage collecting coins
        e.KILLED_OPPONENT: 25,        # Increased reward for killing opponents
        e.INVALID_ACTION: -6,         # Increased penalty for invalid actions
        e.WAITED: -0.5,                 # Increased penalty for waiting
        e.KILLED_SELF: -10,           # Increased penalty for self-destruction
        e.SURVIVED_ROUND: 2,          # Increased reward for survival
        e.CRATE_DESTROYED: 2,         # Reward per crate destroyed
        e.MOVED_DOWN: -0.15,
        e.MOVED_LEFT: -0.15,
        e.MOVED_RIGHT: -0.15,
        e.MOVED_UP: -0.15,
        e.BOMB_DROPPED: 0,            # Base reward for dropping bombs, adjusted below
        e.GOT_KILLED: -10,             # Penalty for being killed
        e.OPPONENT_ELIMINATED: 5,     # Reward for eliminating an opponent
        'MOVED_TO_NEW_POSITION': 2,        # Small reward for exploring
        'MOVED_TO_RECENT_POSITION': -2,      # Penalty for revisiting recent positions
        ##'GAME_WON': 500,                        # Huge reward for winning the game # ONLY USE FOR MULTIPLAYER TRAINING!
        'MOVED_TOWARDS_COIN': 1,
        'MOVED_AWAY_FROM_COIN': -1,
        'MOVED_INTO_BLAST_ZONE': -10,
        'STAYED_IN_BLAST_ZONE': -5,
        'ESCAPED_BLAST_ZONE': 10,
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

    # Compute total reward
    reward = sum(game_rewards.get(event, 0) for event in events)

    # Scale the total reward
    scaling_factor = 0.5
    reward *= scaling_factor

    # Track rewards for logging
    self.rewards_episode.append(reward)
    self.logger.debug(f"Calculated scaled reward: {reward}")

    return reward

def compute_distance_to_nearest_coin(game_state):
    position = game_state['self'][3]
    x, y = position
    coins = game_state['coins']

    if coins:
        distances = [abs(cx - x) + abs(cy - y) for (cx, cy) in coins]
        return min(distances)
    else:
        return None  # No coins left

def is_in_blast_zone(game_state):
    position = game_state['self'][3]
    x, y = position
    arena = game_state['field']

    # Compute bomb map similar to state_to_features
    bomb_map = np.full(arena.shape, np.inf)

    bombs = game_state['bombs']

    for (bx, by), bomb_timer in bombs:
        # Mark the bomb position as dangerous at explosion time
        bomb_map[bx, by] = min(bomb_map[bx, by], bomb_timer)

        for dx, dy in [(-1, 0), (1,0), (0,-1), (0,1)]:
            for i in range(1, 4):
                nx, ny = bx + dx*i, by + dy*i
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    if arena[nx, ny] != 0:
                        break
                    danger_time = bomb_timer - i
                    if danger_time <= 0:
                        bomb_map[nx, ny] = min(bomb_map[nx, ny], danger_time)
                else:
                    break

    # Current position dangerous?
    return bomb_map[x, y] <= 0