import os
import numpy as np
import tensorflow as tf

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    if self.train or not os.path.isfile("my-saved-model.h5"):
        self.logger.info("Setting up model from scratch.")
        # The model will be created in train.py, so we don't need to do anything here
    else:
        self.logger.info("Loading model from saved state.")
        self.model = tf.keras.models.load_model("my-saved-model.h5")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    """
    features = state_to_features(game_state)
    
    if self.train:
        # In training mode, use the act method from the DQNAgent in train.py
        action_index = self.agent.act(features.reshape(1, -1))
    else:
        # In non-training mode, use the loaded model to predict the action
        q_values = self.model.predict(features.reshape(1, -1))
        action_index = np.argmax(q_values[0])
    
    return ACTIONS[action_index]

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.
    """
    if game_state is None:
        return None

    # Extract relevant information from game state
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    (x, y) = game_state['self'][3]
    
    # Initialize feature channels
    channels = []

    # Channel 1: Obstacle map (walls and crates)
    channels.append(np.where(field == -1, 1, 0))  # Walls
    channels.append(np.where(field == 1, 1, 0))   # Crates
    
    # Channel 2: Coin locations
    coin_map = np.zeros_like(field)
    for cx, cy in coins:
        coin_map[cx, cy] = 1
    channels.append(coin_map)
    
    # Channel 3: Bomb locations and timers
    bomb_map = np.zeros_like(field)
    for (bx, by), timer in bombs:
        bomb_map[bx, by] = 1 - timer / 4  # Normalize timer
    channels.append(bomb_map)
    
    # Channel 4: Explosion map
    channels.append(explosion_map)
    
    # Channel 5: Agent's position
    agent_map = np.zeros_like(field)
    agent_map[x, y] = 1
    channels.append(agent_map)
    
    # Channel 6: Danger map (combination of bombs and explosions)
    danger_map = np.maximum(bomb_map, explosion_map)
    channels.append(danger_map)

    # Channel 7: Safe tiles (not wall, not crate, not dangerous)
    safe_map = np.where((field == 0) & (danger_map == 0), 1, 0)
    channels.append(safe_map)

    # Channel 8: Distance to nearest coin (using simple Manhattan distance)
    coin_distance = np.full_like(field, fill_value=np.inf)
    for cx, cy in coins:
        coin_distance = np.minimum(coin_distance, np.abs(np.indices(field.shape) - np.array([[cx], [cy]])[:, None, None]).sum(axis=0))
    coin_distance = np.where(np.isfinite(coin_distance), coin_distance / coin_distance.max(), 0)  # Normalize and handle division by zero
    channels.append(coin_distance)

    # Stack channels and reshape
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def get_nearest_safe_tile(danger_map, x, y):
    """Helper function to find the nearest safe tile."""
    safe_tiles = np.argwhere(danger_map == 0)
    if len(safe_tiles) == 0:
        return None
    distances = np.sum(np.abs(safe_tiles - np.array([x, y])), axis=1)
    nearest = safe_tiles[np.argmin(distances)]
    return tuple(nearest)