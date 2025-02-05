# %%
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

# grid_size = (100, 100)
grid_size = (50, 50)

# positions on the grid
ps = np.linspace(-1, 1, grid_size[0])
# velocities on the grid
vs = np.linspace(-1, 1, grid_size[1])
xs = np.meshgrid(ps, vs)
us = np.linspace(-2, 2, 100)
# us = np.linspace(-2, 2, 20)

final_time = 1.0
dt = 0.1
ts = np.arange(0, final_time+dt, dt)
# running cost matrices
Q = np.eye(2)
R = np.eye(1)


def solve_with_dynamic_programming():
    # the last timestep is needed for the terminal condition.
    values = np.zeros([grid_size[0], grid_size[1], len(ts)+1])
    
    for i in range(len(ts)-1, -1, -1):
        t = ts[i]
        print(f"t={t}")
        # compute the value function at the next timestep
        next_values = values[:, :, i+1]
        interpolator = RegularGridInterpolator(
            (ps, vs),
            next_values,
            bounds_error=False,
            fill_value=None  # Allows extrapolation using nearest-neighbor
        )
        # compute the value function at the current timestep
        current_values = np.zeros_like(next_values)
        for j in range(grid_size[0]):
            for k in range(grid_size[1]):
                p = ps[j]
                v = vs[k]
                state_current = np.array([p, v])
                value_candidates = np.zeros(len(us))
                for (i_u, u) in enumerate(us):
                    # compute the next state
                    p_next = p + dt * v
                    v_next = v + dt * u
                    state_next = np.array([p_next, v_next])
                    # compute the cost
                    cost = (state_current @ Q @ state_current) + (u * R[0, 0] * u)
                    # compute the value function
                    next_value_under_u = interpolator(state_next)
                    cost += next_value_under_u
                    value_candidates[i_u] = cost
                value_min = np.min(value_candidates)
                current_values[j, k] = value_min
        values[:, :, i] = current_values
    return values[:, :, 0]

values_dp = solve_with_dynamic_programming()
# %%
# visualize the value funciton in 3d map
fig = go.Figure(data=[go.Surface(z=values_dp, x=xs[0], y=xs[1])])

# Update layout for better visualization
fig.update_layout(
    title="Value Function at t=0",
    scene=dict(
        xaxis_title="Position (p)",
        yaxis_title="Velocity (v)",
        zaxis_title="Value Function V(p, v)",
    )
)
fig.show()

# %%
def solve_with_riccati_equation():
    A = np.array([[1, 0.1], [0, 1.0]])
    B = np.array([[0], [0.1]])
    P = np.zeros([2, 2])
    for i in range(len(ts)-1, -1, -1):
        K = np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        print(K)
        P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        print(P)
    values = np.zeros(grid_size)
    for j in range(grid_size[0]):
        for k in range(grid_size[1]):
            state = np.array([ps[j], vs[k]])
            values[j, k] = state @ P @ state
    return values

values_riccati = solve_with_riccati_equation()
# Compute error between DP and Riccati solutions
error_values = np.abs(values_dp - values_riccati)
print(f"Max error: {np.max(error_values)}")
print(f"Mean error: {np.mean(error_values)}")
# Create subplots with three 3D surface plots in one row
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Dynamic Programming", "Riccati Solution", "Error (|DP - Riccati|)"),
    specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]]
)

# Add the DP solution
fig.add_trace(
    go.Surface(z=values_dp, x=xs[0], y=xs[1], colorscale="Viridis"),
    row=1, col=1
)

# Add the Riccati solution
fig.add_trace(
    go.Surface(z=values_riccati, x=xs[0], y=xs[1], colorscale="Cividis"),
    row=1, col=2
)

# Add the error plot
fig.add_trace(
    go.Surface(z=error_values, x=xs[0], y=xs[1], colorscale="RdBu", showscale=True),
    row=1, col=3
)

# Update layout for better visualization
fig.update_layout(
    title="Comparison of Value Functions",
    scene=dict(
        xaxis_title="Position (p)",
        yaxis_title="Velocity (v)",
        zaxis_title="Value Function V(p, v)",
    ),
    height=600, width=1800  # Adjust size for better visibility
)

# Show the plot
fig.show()

# %%
