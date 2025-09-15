import numpy as np
from tqdm import tqdm
import pickle
import logging
from numba import jit, prange

# --------------------------------------------------------------------------- #
# --- Numba-JIT Compiled Helper Functions for Performance-Critical Loops ---  #
# --------------------------------------------------------------------------- #

@jit(nopython=True)
def _get_state_indices_numba(cc, cx, T, t, grids_0, grids_1, grids_2, grids_3):
    """Numba-jitted version to find the nearest indices in the grids."""
    idx_cc = np.argmin(np.abs(grids_0 - cc))
    idx_cx = np.argmin(np.abs(grids_1 - cx))
    idx_T = np.argmin(np.abs(grids_2 - T))
    idx_t = np.argmin(np.abs(grids_3 - t))
    return idx_cc, idx_cx, idx_T, idx_t

@jit(nopython=True, parallel=True)
def _precompute_arrival_dynamics_numba(
    R_arrival, P_survival, Next_V_idx_survival,
    grids_tuple, grid_sizes, avg_revenue_per_cx, failure_cost,
    replacement_cost, cum_baseline_values
):
    """
    Fills the dynamics arrays for the arrival state using parallel loops.
    This version uses a pre-computed expected revenue for each cx bin.
    """
    N_cc, N_cx, N_T, N_t = grid_sizes
    grids_0, grids_1, grids_2, grids_3 = grids_tuple

    for idx_cc in prange(N_cc):
        for idx_cx in range(N_cx):
            for idx_T in range(N_T):
                for idx_t in range(N_t):
                    cc, cx, T, t = grids_0[idx_cc], grids_1[idx_cx], grids_2[idx_T], grids_3[idx_t]

                    # **MODIFICATION**: Look up pre-computed expected revenue
                    revenue = avg_revenue_per_cx[idx_cx]
                    exp_term = np.exp(cc + cx)

                    t_plus_T = t + T
                    idx_t_next_lookup = np.argmin(np.abs(grids_3 - t_plus_T))
                    delta_max = cum_baseline_values[idx_t_next_lookup] - cum_baseline_values[idx_t]
                    max_incremental_hazard = exp_term * delta_max
                    
                    p_survival = np.exp(-max_incremental_hazard)
                    P_survival[idx_cc, idx_cx, idx_T, idx_t] = p_survival

                    R_surv = revenue
                    R_fail = revenue - failure_cost - replacement_cost
                    R_arrival[idx_cc, idx_cx, idx_T, idx_t] = p_survival * R_surv + (1 - p_survival) * R_fail

                    cc_next_surv, t_next_surv = cc + cx, t + T
                    idx_cc_next_s, _, _, idx_t_next_s = _get_state_indices_numba(
                        cc_next_surv, 0.0, 0.0, t_next_surv,
                        grids_0, grids_1, grids_2, grids_3
                    )
                    Next_V_idx_survival[idx_cc, idx_cx, idx_T, idx_t, 0] = idx_cc_next_s
                    Next_V_idx_survival[idx_cc, idx_cx, idx_T, idx_t, 1] = idx_t_next_s

@jit(nopython=True, parallel=True)
def _compute_expected_V_arrival_numba(V_arrival, sampled_customer_indices_arr, N_cc, N_t):
    """Computes the expected value of the next 'arrival' state."""
    num_samples = sampled_customer_indices_arr.shape[0]
    expected_V = np.zeros((N_cc, N_t))
    
    for i in prange(N_cc):
        for j in range(N_t):
            total_value = 0.0
            for k in range(num_samples):
                idx_cx, idx_T = sampled_customer_indices_arr[k, 0], sampled_customer_indices_arr[k, 1]
                total_value += V_arrival[i, idx_cx, idx_T, j]
            expected_V[i, j] = total_value / num_samples
    return expected_V

# --------------------------------------------------------------------------- #
# ---                         Main Agent Class                            --- #
# --------------------------------------------------------------------------- #

class DiscretizedDPAgent:
    """
    Solves the MDP using value iteration on a discretized state space.
    This version correctly handles revenue calculation via expectation.
    """
    def __init__(self, N, max_cumulative_context, max_active_time, u_hat, degradation_learner, customer_generator, params):
        self.degradation_learner = degradation_learner
        self.customer_generator = customer_generator
        self.params = params
        self.gamma = params.get('gamma', 0.99)
        self.theta = self.degradation_learner.get_theta()
        self.u_hat = u_hat # **MODIFICATION**: Store u_hat for revenue calculation
        
        self._setup_discretization(N, max_cumulative_context, max_active_time)

        self.V_arrival = np.zeros(self.grid_shape)
        self.V_departure = np.zeros((self.grid_sizes[0], self.grid_sizes[3]))
        self.policy_arrival = np.zeros(self.grid_shape, dtype=int)
        self.policy_departure = np.zeros(self.V_departure.shape, dtype=int)

        self._precompute_dynamics()
    
    # ... `_setup_discretization` and `_get_state_indices` methods are unchanged ...
    def _setup_discretization(self, N, max_cumulative_context, max_active_time):
        """Creates the grids for each dimension of the state space."""
        if isinstance(N, int):
            N = [N, N, N, N]
        
        duration_lambda = self.params.get('duration_lambda', 1.0)
        max_rental_duration = -np.log(0.0005) * duration_lambda
        max_customer_context = 1.0

        self.grid_max_vals = [max_cumulative_context, max_customer_context, max_rental_duration, max_active_time]
        self.grid_sizes = N
        self.grid_shape = tuple(N)
        
        self.grids = [
            np.linspace(0, self.grid_max_vals[0], self.grid_sizes[0]),
            np.linspace(0, self.grid_max_vals[1], self.grid_sizes[1]),
            np.linspace(0, self.grid_max_vals[2], self.grid_sizes[2]),
            np.linspace(0, self.grid_max_vals[3], self.grid_sizes[3])
        ]
        
        print("Discretization setup:")
        print(f"  - Cumulative Context: {self.grid_sizes[0]} steps up to {self.grid_max_vals[0]:.2f}")
        print(f"  - Customer Context:   {self.grid_sizes[1]} steps up to {self.grid_max_vals[1]:.2f}")
        print(f"  - Rental Duration:    {self.grid_sizes[2]} steps up to {self.grid_max_vals[2]:.2f} (99.95th percentile)")
        print(f"  - Active Time:        {self.grid_sizes[3]} steps up to {self.grid_max_vals[3]:.2f}")

    def _get_state_indices(self, state_values):
        """(Python-version) Finds the nearest indices for a given continuous state vector."""
        cc, cx, T, t = state_values
        idx_cc = np.argmin(np.abs(self.grids[0] - cc))
        idx_cx = np.argmin(np.abs(self.grids[1] - cx))
        idx_T = np.argmin(np.abs(self.grids[2] - T))
        idx_t = np.argmin(np.abs(self.grids[3] - t))
        return idx_cc, idx_cx, idx_T, idx_t

    def _precompute_dynamics(self, num_samples=10000):
        """
        Pre-computes dynamics, including the expected revenue for each cx bin.
        """
        print(f"Pre-computing expectations from {num_samples} customer samples...")
        
        # --- Step 1: Sample customers and build the cx -> x mapping ---
        cx_to_x_map = [[] for _ in range(self.grid_sizes[1])]
        sampled_customer_indices = []
        for _ in range(num_samples):
            customer = self.customer_generator.generate()
            x, duration = customer['context'], customer['desired_duration']
            customer_context_val = np.dot(self.theta, x)
            _, idx_cx, idx_T, _ = self._get_state_indices((0, customer_context_val, duration, 0))
            sampled_customer_indices.append((idx_cx, idx_T))
            cx_to_x_map[idx_cx].append(x) # Store the original x vector

        self.sampled_customer_indices_arr = np.array(sampled_customer_indices, dtype=np.int32)
        
        # --- Step 2: Compute the expected revenue for each cx bin ---
        avg_revenue_per_cx = np.zeros(self.grid_sizes[1])
        for i, x_list in enumerate(cx_to_x_map):
            if x_list: # If the bin is not empty
                revenues = [np.dot(self.u_hat, x) for x in x_list]
                avg_revenue_per_cx[i] = np.mean(revenues)
        
        # --- Step 3: Handle empty bins using interpolation ---
        non_empty_indices = np.where(avg_revenue_per_cx > 0)[0]
        if len(non_empty_indices) == 0:
            print("Warning: No customers sampled. Revenue will be zero.")
        elif len(non_empty_indices) < 2:
            avg_revenue_per_cx.fill(avg_revenue_per_cx[non_empty_indices[0]])
        else:
            all_indices = np.arange(self.grid_sizes[1])
            avg_revenue_per_cx = np.interp(all_indices, non_empty_indices, avg_revenue_per_cx[non_empty_indices])

        # --- Step 4: Call Numba pre-computation for the rest of the dynamics ---
        interarrival_lambda = self.params.get('interarrival_lambda', 1.0)
        self.expected_holding_reward = -self.params['holding_cost_rate'] * self.params['interarrival_lambda'] # this is actually inverse lambda (aka scale)
        
        self.R_arrival_expected = np.zeros(self.V_arrival.shape)
        self.P_survival = np.zeros(self.V_arrival.shape)
        self.Next_V_idx_survival = np.zeros((*self.V_arrival.shape, 2), dtype=np.int32)
        cum_baseline_values = self.degradation_learner.cum_baseline(self.grids[3])

        print("Starting Numba-accelerated pre-computation of arrival dynamics...")
        _precompute_arrival_dynamics_numba(
            self.R_arrival_expected, self.P_survival, self.Next_V_idx_survival,
            tuple(self.grids), self.grid_shape, avg_revenue_per_cx,
            self.params['failure_cost'], self.params['replacement_cost'], 
            cum_baseline_values
        )
        print("Pre-computation complete. ✅")
    
    # ... `run_value_iteration`, `get_policy`, `save_policy`, and `load_policy` are unchanged ...
    def run_value_iteration(self, num_iterations, tolerance=1e-4):
        """Performs value iteration, using Numba to accelerate expectation calculations."""
        print("\nStarting Value Iteration...")
        history = {'delta': []}
        for i in range(num_iterations):
            # --- 1. Update Departure State Values ---
            V_departure_old = self.V_departure.copy()
            
            # Use Numba-accelerated function for the expensive expectation calculation
            expected_V_arrival = _compute_expected_V_arrival_numba(
                self.V_arrival, self.sampled_customer_indices_arr,
                self.grid_sizes[0], self.grid_sizes[3]
            )
            
            q_replace_value = self.expected_holding_reward - self.params['replacement_cost'] + self.gamma * expected_V_arrival[0, 0]
            q_replace = np.full(self.V_departure.shape, q_replace_value)
            
            q_no_replace = self.expected_holding_reward + self.gamma * expected_V_arrival
            self.V_departure = np.maximum(q_replace, q_no_replace)
            self.policy_departure = np.argmax(np.stack([q_replace, q_no_replace], axis=-1), axis=-1) + 2

            # --- 2. Update Arrival State Values (already vectorized) ---
            V_arrival_old = self.V_arrival.copy()
            
            V_next_shutdown = self.V_departure.reshape(self.grid_sizes[0], 1, 1, self.grid_sizes[3])
            q_shutdown = 0 + self.gamma * V_next_shutdown

            V_next_survival = self.V_departure[self.Next_V_idx_survival[..., 0], self.Next_V_idx_survival[..., 1]]
            V_next_failure = self.V_departure[0, 0]
            
            expected_V_next_give_price = self.P_survival * V_next_survival + (1 - self.P_survival) * V_next_failure
            q_give_price = self.R_arrival_expected + self.gamma * expected_V_next_give_price

            self.V_arrival = np.maximum(q_give_price, q_shutdown)
            q_shutdown_broadcasted = np.broadcast_to(q_shutdown, q_give_price.shape)
            self.policy_arrival = np.argmin(np.stack([q_give_price, q_shutdown_broadcasted], axis=-1), axis=-1)

            # --- 3. Check for Convergence ---
            delta = max(
                np.max(np.abs(self.V_arrival - V_arrival_old)),
                np.max(np.abs(self.V_departure - V_departure_old))
            )
            history['delta'].append(delta)
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{num_iterations} | Max Change (Delta): {delta:.6f}")
            
            if delta < tolerance:
                print(f"\nValue iteration converged after {i+1} iterations. ✨")
                break
        
        if i == num_iterations -1:
            print("\nValue iteration finished (max iterations reached).")
        return history

    def get_policy(self, nothing):
        """Returns a function that represents the learned greedy policy."""
        def policy_fn(state, policy_kwargs):
            """
            Args:
                state (tuple): (cum_context, cust_context, duration, active_time, phase).
            """
            cc, cx, T, t, phase = state
            indices = self._get_state_indices((cc, cx, T, t))
            
            if phase == 0: # Arrival
                return self.policy_arrival[indices]
            else: # Departure
                idx_cc, _, _, idx_t = indices
                return self.policy_departure[idx_cc, idx_t]
        return policy_fn

    def save_policy(self, filepath):
        """Saves the essential policy components to a file."""
        policy_data = {
            'grids': self.grids,
            'policy_arrival': self.policy_arrival,
            'policy_departure': self.policy_departure,
            'params': self.params
        }
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
        print(f"Policy saved to {filepath}")

    @classmethod
    def load_policy(cls, filepath):
        """Loads a policy and returns a callable policy function."""
        with open(filepath, 'rb') as f:
            policy_data = pickle.load(f)
        
        grids = policy_data['grids']
        policy_arrival = policy_data['policy_arrival']
        policy_departure = policy_data['policy_departure']

        @jit(nopython=True) # JIT the lookup function for faster policy execution
        def _get_indices_fast(cc, T, t, grids_0, grids_2, grids_3):
            # cx is not needed for departure policy
            idx_cc = np.argmin(np.abs(grids_0 - cc))
            idx_T = np.argmin(np.abs(grids_2 - T))
            idx_t = np.argmin(np.abs(grids_3 - t))
            return idx_cc, idx_T, idx_t

        def policy_fn(state):
            cc, cx, T, t, phase = state
            if phase == 0: # Arrival
                # For single lookups, Python version is fine
                indices = (np.argmin(np.abs(grids[0] - cc)), np.argmin(np.abs(grids[1] - cx)),
                           np.argmin(np.abs(grids[2] - T)), np.argmin(np.abs(grids[3] - t)))
                return policy_arrival[indices]
            else: # Departure
                # A specialized faster lookup could be used here if needed
                idx_cc, _, _, idx_t = (np.argmin(np.abs(grids[0] - cc)), 0, 0, np.argmin(np.abs(grids[3] - t)))
                return policy_departure[idx_cc, idx_t]
        
        print(f"Policy loaded from {filepath}")
        return policy_fn