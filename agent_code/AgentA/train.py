from collections import namedtuple, deque
import numpy as np
import pickle
import random
import tensorflow as tf
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS  # Import ACTIONS from callbacks

# Named tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 10000  # Replay memory size
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQUENCY = 10  # How often to update the target network
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=TRANSITION_HISTORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def setup_training(self):
    """
    Initialize self for training purpose.
    """
    # Create a dummy game state for feature size calculation
    dummy_game_state = {
        'field': np.zeros((17, 17)),  # Assuming a 17x17 board
        'bombs': [],
        'explosion_map': np.zeros((17, 17)),
        'coins': [],
        'self': ('agent_name', 0, True, (0, 0)),  # Agent at position (0, 0)
        'others': []
    }

    # Get the size of the state vector (number of features)
    state_size = len(state_to_features(dummy_game_state))
    action_size = len(ACTIONS)

    self.agent = DQNAgent(state_size, action_size)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Transform state and next state into feature vectors
    state = state_to_features(old_game_state)
    next_state = state_to_features(new_game_state)

    # Convert action from string to an index
    action = ACTIONS.index(self_action)

    # Get the reward
    reward = reward_from_events(self, events)

    # Remember the transition
    self.agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1) if next_state is not None else None)

    # Perform a replay to learn from past experiences
    self.agent.replay()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # Process the final state transition
    state = state_to_features(last_game_state)
    action = ACTIONS.index(last_action)
    reward = reward_from_events(self, events)
    
    self.agent.remember(state.reshape(1, -1), action, reward, None)
    
    # Replay experience to train the agent
    self.agent.replay()

    # Update the target network every few rounds
    if self.rounds % TARGET_UPDATE_FREQUENCY == 0:
        self.agent.update_target_model()

    # Save the model
    self.agent.save("my-saved-model.h5")

    self.rounds += 1

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets based on the events.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,                      # Reward for collecting a coin
        e.KILLED_OPPONENT: 10,                    # Increased reward for killing an opponent
        e.CRATE_DESTROYED: 2,                     # Reward for destroying a crate
        e.MOVED_TOWARD_COIN: 0.5,                 # Reward for moving closer to a coin
        e.MOVED_AWAY_FROM_DANGER: 0.5,            # Reward for moving away from danger
        e.SURVIVED_ROUND: 5,                      # Reward for surviving the round
        e.INVALID_ACTION: -1,                     # Penalty for making an invalid move
        e.KILLED_SELF: -10,                       # Large penalty for killing self
        e.MOVED_TOWARD_SAFE_TILE: 0.5,            # Reward for moving toward a safe tile
        e.MOVED_AWAY_FROM_EXPLOSION: 1,           # Reward for moving away from an explosion
        e.MOVED_TOWARD_ENEMY: -0.5,               # Slight penalty for moving toward an enemy (unless attacking)
        e.MOVED_TOWARD_BOMB: -1,                  # Penalty for moving closer to an active bomb
        e.PLACED_BOMB_STRATEGICALLY: 2,           # Reward for placing a bomb in a good position (e.g., near multiple crates or opponents)
        PLACEHOLDER_EVENT: -0.1                   # Example of a custom event being bad
    }

    reward_sum = sum([game_rewards.get(event, 0) for event in events])
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
