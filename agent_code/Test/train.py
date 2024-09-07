from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
EXPLORED_NEW_POSITION = "EXPLORED_NEW_POSITION"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    # Load or initialize Q-table
    try:
        with open("q-table.pt", "rb") as file:
            self.q_table = pickle.load(file)
        print("Q-table loaded successfully.")
    except (FileNotFoundError, EOFError):
        self.q_table = {}
        print("New Q-table initialized.")
    
    # Initialize performance tracking
    self.scores = []
    self.games_played = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Add custom events
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    
    if old_features is not None and new_features is not None:
        if new_features[0] < old_features[0]:  # coin distance decreased
            events.append(MOVED_TOWARDS_COIN)
        elif new_features[0] > old_features[0]:  # coin distance increased
            events.append(MOVED_AWAY_FROM_COIN)
        if new_features[8] > old_features[8]:  # exploration ratio increased
            events.append(EXPLORED_NEW_POSITION)

    # Calculate reward
    reward = reward_from_events(self, events)

    # Store transition
    self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Q-learning update
    if old_features is not None:
        old_state = tuple(old_features)
        new_state = tuple(new_features)

        # Initialize Q-values if not present
        if old_state not in self.q_table:
            self.q_table[old_state] = {action: 0 for action in ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']}

        # Q-learning update formula
        old_q_value = self.q_table[old_state][self_action]
        if new_state in self.q_table:
            max_future_q = max(self.q_table[new_state].values())
        else:
            max_future_q = 0
        
        new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
        self.q_table[old_state][self_action] = new_q_value

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Add the last transition
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_features, last_action, None, reward))

    # Perform Q-learning update for the last step
    if last_features is not None:
        last_state = tuple(last_features)
        if last_state not in self.q_table:
            self.q_table[last_state] = {action: 0 for action in ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']}
        
        old_q_value = self.q_table[last_state][last_action]
        new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * reward
        self.q_table[last_state][last_action] = new_q_value

    # Store the Q-table
    with open("q-table.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    # Increment games played and store performance metrics
    self.games_played += 1
    score = last_game_state['self'][1]
    self.scores.append(score)
    
    # Create performance graph every 100 games
    create_performance_graph(self)

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets to encourage certain behavior.
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
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def create_performance_graph(self):
    if self.games_played % 100 == 0 and self.games_played > 0:
        plt.figure(figsize=(12, 6))
        
        # Plot individual scores
        plt.plot(range(1, self.games_played + 1), self.scores, label='Score', alpha=0.5)
        
        # Calculate and plot moving average
        moving_avg = np.convolve(self.scores, np.ones(100)/100, mode='valid')
        plt.plot(range(100, self.games_played + 1), moving_avg, label='Moving Average (100 rounds)', color='red')
        
        plt.title(f'Agent Performance over {self.games_played} Games')
        plt.xlabel('Game Number')
        plt.ylabel('Score')
        plt.legend()
        
        # Adjust y-axis to start from 0
        plt.ylim(bottom=0)
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.savefig(f'performance_graph_{self.games_played}.png')
        plt.close()