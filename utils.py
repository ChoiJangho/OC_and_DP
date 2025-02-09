from scipy.interpolate import RegularGridInterpolator
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def solve_with_dynamic_programming(ps, vs, us, ts, Q, R, grid_size, dt):
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

def solve_with_riccati_equation(ps, vs, ts, Q, R, grid_size):
    A = np.array([[1, 0.1], [0, 1.0]])
    B = np.array([[0], [0.1]])
    P = np.zeros([2, 2])
    Ks = np.zeros([len(ts), 1, 2])
    for i in range(len(ts)-1, -1, -1):
        K = np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        Ks[i, 0, :] = K
        P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
    values = np.zeros(grid_size)
    for j in range(grid_size[0]):
        for k in range(grid_size[1]):
            state = np.array([ps[j], vs[k]])
            values[j, k] = state @ P @ state
    return values, Ks

def simulate_lqr(x0, ts, Ks, dt):
    states = [x0]
    state = x0
    for i in range(len(ts)-1):
        u = -Ks[i, 0, :] @ state
        state = state + dt * (np.array([state[1], u]))
        states.append(state)
    return np.asarray(states)


def visualize_ball_motion(ps, vs):
    """
    Visualizes the motion of a ball along a 1D axis with transparency and velocity arrows.

    Parameters:
    - ps: Array of ball positions over time.
    - vs: Array of ball velocities over time.
    """

    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot ground
    ax.axhline(y=-0.1, color='k', linewidth=2)  # Shift ground slightly below the balls

    # Define transparency gradient from 0.2 to 1.0
    transparencies = np.linspace(0.1, 0.5, len(ps))

    # Plot ball positions as circles with transparency gradient
    ball_radius = 0.1
    for i in range(len(ps)):
        ax.add_patch(plt.Circle((ps[i], ball_radius), ball_radius, color='b', alpha=transparencies[i]))

    # Plot velocity as arrows starting from the center of the balls
    for i in range(len(ps)):
        if abs(vs[i]) < 1e-6:
            continue
        ax.arrow(ps[i], ball_radius, vs[i] * 0.4, 0, head_width=0.02, head_length=0.02, fc='r', ec='r')

    # Set plot limits and labels
    ax.set_xlim(min(ps) - 2 * ball_radius, max(ps) + 2 * ball_radius)
    ax.set_ylim(0.0, 3 * ball_radius)
    ax.set_xlabel("Position")
    ax.set_yticks([])  # Remove y-axis ticks for cleaner visualization
    ax.set_aspect('equal')

    plt.show()

def visualize_value_function(xs, values):
    fig = go.Figure(data=[go.Surface(z=values, x=xs[0], y=xs[1])])
    # Update layout for better visualization
    fig.update_layout(
        title="Value Function at t=0",
        scene=dict(
            xaxis_title="Position (p)",
            yaxis_title="Velocity (v)",
            zaxis_title="Value Function V(p, v)",
        )
    )
    return fig

def visualize_value_functions_for_comparison(xs, values_dp, values_riccati):
    error_values = np.abs(values_dp - values_riccati)
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
    fig.update_layout(
        scene=dict(zaxis=dict(range=[0, 30])),
        scene2=dict(zaxis=dict(range=[0, 30])),
        scene3=dict(zaxis=dict(range=[0, 30]))
    )
    fig.update_layout(
        title="Comparison of Value Functions",
        scene=dict(
            xaxis_title="Position (p)",
            yaxis_title="Velocity (v)",
            zaxis_title="Value Function V(p, v)",
        ),
        height=600, width=1800
    )
    return fig