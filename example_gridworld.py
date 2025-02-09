# %%
import numpy as np

# Define gridworld dimensions and parameters
GRID_SIZE = 3
GOAL_STATE = (0, 2)  # (1,1) in 0-indexed grid
PENALTY_STATE1 = (1, 2)  # (2,1) in 0-indexed grid
PENALTY_STATE2 = (1, 1)
REWARD_GOAL = 1
REWARD_PENALTY = -10
TIME_HORIZON = 5
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # Up, Down, Left, Right
ACTIONS_STRING = ["Up", "Down", "Left", "Right", "Stay"]

# Initialize value function
value_function = np.zeros((GRID_SIZE, GRID_SIZE, TIME_HORIZON + 1))

# Transition function
def next_state(state, action):
    x, y = state
    dx, dy = action
    nx, ny = x + dx, y + dy
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
        return nx, ny
    return x, y  # Stay in place if moving out of bounds

# Reward function
def reward(state):
    if state == GOAL_STATE:
        return REWARD_GOAL
    elif state == PENALTY_STATE1 or state == PENALTY_STATE2:
        return REWARD_PENALTY
    else:
        return 0

# Solve using dynamic programming
for t in range(TIME_HORIZON-1, -1, -1):  # Work backward from T-1 to 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            state = (x, y)
            max_value = float('-inf')
            for action in ACTIONS:
                next_s = next_state(state, action)
                # Bellman equation
                current_value = reward(state) + value_function[next_s[0], next_s[1], t + 1]
                max_value = max(max_value, current_value)
            value_function[x, y, t] = max_value
    print(f"Value Function at t={t}:")
    print(value_function[:, :, t])

# Extract optimal policy
policy = np.full((GRID_SIZE, GRID_SIZE), None)

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        state = (x, y)
        best_action_index = None
        max_value = float('-inf')
        for (i, action) in enumerate(ACTIONS):
            next_s = next_state(state, action)
            if next_s == state and action != (0, 0):
                continue
            current_value = reward(state) + value_function[next_s[0], next_s[1], 1]
            if current_value > max_value:
                max_value = current_value
                best_action = ACTIONS_STRING[i]
        policy[x, y] = best_action

print("\nOptimal Policy:")
for x in range(GRID_SIZE):
    print([policy[x, y] for y in range(GRID_SIZE)])


# %%
