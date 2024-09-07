import os
import pickle
import random
import numpy as np
from collections import deque


# Add this as a global variable at the top of your file
position_history = deque(maxlen=2000)  # Adjust maxlen as needed

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Q-learning parameters (should match those in train.py)
EPSILON = 0.1

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("q-table.pt"):
        self.logger.info("Setting up model from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("q-table.pt", "rb") as file:
            self.q_table = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    if features is not None:
        state = tuple(features)
        if self.train and random.random() < EPSILON:
            # Explore: choose a random action
            self.logger.debug("Choosing action purely at random.")
            return np.random.choice(ACTIONS)
        else:
            # Exploit: choose the best action according to Q-table
            self.logger.debug("Querying model for action.")
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return np.random.choice(ACTIONS)
    else:
        self.logger.debug("Game state is None, choosing random action.")
        return np.random.choice(ACTIONS)



def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    global position_history

    # This is the dict before the game begins and after it ends
    if game_state is None:
        position_history.clear()
        return None

    # Extract relevant information from game state
    arena = game_state['field']
    coins = game_state['coins']
    (name, score, bomb_possible, (x, y)) = game_state['self']

    # Initialize feature list
    features = []

    # Feature 1: Distance to nearest coin
    if coins:
        distances = [manhattan_distance((x, y), coin) for coin in coins]
        nearest_coin_distance = min(distances)
        features.append(nearest_coin_distance)
    else:
        features.append(-1)  # No coins left

    # Feature 2-3: Direction to nearest coin
    if coins:
        nearest_coin = coins[distances.index(nearest_coin_distance)]
        coin_direction = (nearest_coin[0] - x, nearest_coin[1] - y)
        features.extend(coin_direction)
    else:
        features.extend((0, 0))

    # Feature 4: Number of coins left
    features.append(len(coins))

    # Feature 5-8: Can move in each direction
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # up, right, down, left
        features.append(int(arena[x+dx, y+dy] == 0))

    # Update position history
    position_history.append((x, y))

    # Feature 9: Steps in current position
    steps_in_place = 1
    for past_pos in reversed(list(position_history)[:-1]):
        if past_pos == (x, y):
            steps_in_place += 1
        else:
            break
    features.append(steps_in_place)

    # Feature 10: Exploration ratio
    unique_positions = len(set(position_history))
    total_steps = len(position_history)
    exploration_ratio = unique_positions / total_steps if total_steps > 0 else 0
    features.append(exploration_ratio)

    return np.array(features)


def manhattan_distance(pos1, pos2):
    return abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1]))