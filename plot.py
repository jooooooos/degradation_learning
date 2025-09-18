import numpy as np
import numba
from tqdm import tqdm
import matplotlib.pyplot as plt

# This is the new, high-performance core function compiled by Numba.
# It contains the logic from the original nested loops.
@numba.njit(parallel=True, fastmath=True)
def _compute_grid_numba(duration_grid,
                        cum_context_grid,
                        total_context_grid,
                        sampled_cx,
                        sampled_ux,
                        policy_arrival,
                        grids_tuple):
    """
    Numba-accelerated core function to compute the acceptance grid.
    This function should not be called directly.
    """
    grid_res = len(duration_grid)
    n_samples = len(sampled_cx)
    policy_dims = policy_arrival.shape

    acceptance_count_grid = np.zeros((grid_res, grid_res))
    acceptance_total_grid = np.zeros((grid_res, grid_res))

    total_context_bins = (total_context_grid[:-1] + total_context_grid[1:]) / 2.0

    for j in numba.prange(grid_res):
        duration = duration_grid[j]
        for i in range(grid_res):
            cum_context = cum_context_grid[i]

            for s in range(n_samples):
                cx = sampled_cx[s]
                ux = sampled_ux[s]
                total_context = cum_context + cx

                # --- FIXED SECTION ---
                # Replaced np.clip with min() and max() for scalar values.
                raw_idx0 = np.digitize(total_context, grids_tuple[0]) - 1
                idx0 = max(0, min(raw_idx0, policy_dims[0] - 1))

                raw_idx1 = np.digitize(cx, grids_tuple[1]) - 1
                idx1 = max(0, min(raw_idx1, policy_dims[1] - 1))

                raw_idx2 = np.digitize(ux, grids_tuple[2]) - 1
                idx2 = max(0, min(raw_idx2, policy_dims[2] - 1))

                raw_idx3 = np.digitize(duration, grids_tuple[3]) - 1
                idx3 = max(0, min(raw_idx3, policy_dims[3] - 1))
                # --- END OF FIX ---

                action = policy_arrival[idx0, idx1, idx2, idx3]

                ind = np.searchsorted(total_context_bins, total_context)

                if action == 0:
                    acceptance_count_grid[j, ind] += 1
                acceptance_total_grid[j, ind] += 1

    return acceptance_count_grid, acceptance_total_grid



def compute_arrival_acceptance_probability_accelerated(ddpa, customer_generator, utility, n_samples=500, grid_resolution=50):
    """
    Computes the probability of accepting a job for a grid of total degradation vs. duration.
    This version is accelerated with Numba.
    """
    print("Pre-sampling customer contexts...")
    sampled_customers = [customer_generator.generate() for _ in range(n_samples)]
    sampled_cx = np.array([np.dot(ddpa.theta, c['context']) for c in sampled_customers])
    sampled_ux = np.array([np.dot(utility, c['context']) for c in sampled_customers])

    print("Defining heatmap grid...")
    max_total_context = ddpa.grid_max_vals[0] + 1.0
    total_context_grid = np.linspace(0, max_total_context, grid_resolution)
    cum_context_grid = np.linspace(0, ddpa.grid_max_vals[0], grid_resolution)
    duration_grid = np.linspace(0, ddpa.grid_max_vals[3], grid_resolution)

    # Extract NumPy arrays from the ddpa object.
    # Numba works with simple data types, not complex Python objects.
    policy = ddpa.policy_arrival
    # Numba requires a tuple of arrays, not a list.
    grids_tuple = tuple(g for g in ddpa.grids)

    print("Running accelerated computation...")
    acceptance_count_grid, acceptance_total_grid = _compute_grid_numba(
        duration_grid,
        cum_context_grid,
        total_context_grid,
        sampled_cx,
        sampled_ux,
        policy,
        grids_tuple
    )
    print("Computation finished.")

    # Safely compute the probability, avoiding division by zero.
    acceptance_prob_grid = np.zeros_like(acceptance_count_grid, dtype=np.float64)
    np.divide(acceptance_count_grid, acceptance_total_grid, out=acceptance_prob_grid, where=acceptance_total_grid != 0)

    return acceptance_prob_grid, total_context_grid, duration_grid

# --- How to Use the Accelerated Function ---

# The plotting function remains the same
def plot_acceptance_probability_heatmap(prob_grid, context_grid, duration_grid):
    """
    Plots the computed acceptance probability grid as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(prob_grid, origin='lower', aspect='auto', cmap='viridis',
                   extent=[context_grid[0], context_grid[-1],
                           duration_grid[0], duration_grid[-1]])
    ax.set_xlabel(r'Total Degradation Context ($\theta^T(X+x)$)', fontsize=14)
    ax.set_ylabel('Desired Rental Duration (T)', fontsize=14)
    ax.set_title('Probability of Job Acceptance', fontsize=16, pad=20)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Acceptance Probability', fontsize=12)
    plt.show()


# Assume 'perfect_dpagent', 'customer_gen', and 'UTILITY_TRUE' are defined elsewhere

# 1. Compute the probability data using the FAST version
# prob_data, context_axis, duration_axis = compute_arrival_acceptance_probability_accelerated(
#     perfect_dpagent,
#     customer_gen,
#     UTILITY_TRUE,
#     n_samples=100000,
#     grid_resolution=75
# )

# 2. Plot the resulting heatmap
# plot_acceptance_probability_heatmap(prob_data, context_axis, duration_axis)       