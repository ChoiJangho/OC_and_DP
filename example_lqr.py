# %%
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

import time
import utils

## Problem setup
grid_size = (100, 100)
# grid_size = (50, 50)

# positions on the grid
ps = np.linspace(-1, 1, grid_size[0])
# velocities on the grid
vs = np.linspace(-1, 1, grid_size[1])
xs = np.meshgrid(ps, vs)
# ctrl (acceleration)
us = np.linspace(-2, 2, 100)
# us = np.linspace(-2, 2, 20)

final_time = 1.0
dt = 0.1
ts = np.arange(0, final_time+dt, dt)
# running cost matrices
Q = np.eye(2)
R = np.eye(1)

## Method 1: Solve with dynamic programming
start_dp_time = time.process_time()
values_dp = utils.solve_with_dynamic_programming(ps, vs, us, ts, Q, R, grid_size, dt)
end_dp_time = time.process_time()
dp_time = end_dp_time - start_dp_time
# visualize the value funciton in 3d map
fig = utils.visualize_value_function(xs, values_dp)
# fig.show()

## Method 2: Solve with Riccati equation
start_riccati_time = time.process_time()
values_riccati, Ks = utils.solve_with_riccati_equation(ps, vs, ts, Q, R, grid_size)
end_riccati_time = time.process_time()
riccati_time = end_riccati_time - start_riccati_time
# Compute error between DP and Riccati solutions
error_values = np.abs(values_dp - values_riccati)
print(f"Max error: {np.max(error_values)}")
print(f"Mean error: {np.mean(error_values)}")
print(f"Solve time: DP={dp_time}, Riccati={riccati_time}")
# visualize two value functions side by side and the difference
fig = utils.visualize_value_functions_for_comparison(xs, values_dp, values_riccati)
# fig.show()

# %%
## Simulate the LQR controller.
# simulate for 5 second
ts = np.arange(0, 5.0, dt)
state_initial = np.array([1.0, 0.0])

values_riccati, Ks = utils.solve_with_riccati_equation(ps, vs, ts, Q, R, grid_size)
states = utils.simulate_lqr(state_initial, ts, Ks, dt)
utils.visualize_ball_motion(states[:, 0], states[:, 1])

# Test different Q, R
Q = np.array([[10, 0], [0, 10]])
R = np.array([[1]])
values_riccati, Ks = utils.solve_with_riccati_equation(ps, vs, ts, Q, R, grid_size)
states = utils.simulate_lqr(state_initial, ts, Ks, dt)
utils.visualize_ball_motion(states[:, 0], states[:, 1])

# %%
