"""
Optimized discrete policy via value iteration.

Drop-in replacement for discrete_policy.py with the following optimizations:
  1. Exploit separability: P_survival is 4D (cc,cx,T,t), indices are 2D
  2. Compute 4D base_value in VI loop, broadcast with cu for 5D result
  3. Replace 100K-sample loop with vectorized gather + einsum
  4. Defer policy computation to after convergence
  5. Double-buffer V_arrival to eliminate copies
  6. float32 for all value arrays (halves memory, 2x SIMD throughput)
  7. Fused numba kernel for the arrival update (eliminates all temporaries)
"""

import numpy as np
from collections import Counter
import pickle
from numba import jit, prange
import logging

PARALLEL = True  # Set False if numba parallel causes issues on Apple Silicon


# --------------------------------------------------------------------------- #
# ---                        Numba Kernels                                --- #
# --------------------------------------------------------------------------- #

@jit(nopython=True)
def _get_state_indices_numba(cc, cx, cu, T, t, grids_0, grids_1, grids_2, grids_3, grids_4):
    """Numba-jitted version to find the nearest indices in the grids."""
    idx_cc = np.argmin(np.abs(grids_0 - cc))
    idx_cx = np.argmin(np.abs(grids_1 - cx))
    idx_cu = np.argmin(np.abs(grids_2 - cu))
    idx_T = np.argmin(np.abs(grids_3 - T))
    idx_t = np.argmin(np.abs(grids_4 - t))
    return idx_cc, idx_cx, idx_cu, idx_T, idx_t


@jit(nopython=True, parallel=PARALLEL)
def _precompute_P_survival_4D(P_survival_4D, grids_0, grids_1, grids_3, grids_4,
                               N_cc, N_cx, N_T, N_t, Delta_Lambda):
    """Compute survival probabilities — depends on (cc, cx, T, t) only, NOT cu."""
    for idx_cc in prange(N_cc):
        cc = grids_0[idx_cc]
        for idx_cx in range(N_cx):
            cx = grids_1[idx_cx]
            exp_cc_cx = np.exp(cc + cx)
            for idx_T in range(N_T):
                for idx_t in range(N_t):
                    incremental_hazard = Delta_Lambda[idx_t, idx_T] * exp_cc_cx
                    P_survival_4D[idx_cc, idx_cx, idx_T, idx_t] = np.exp(-incremental_hazard)


@jit(nopython=True)
def _precompute_next_cc_idx(Next_cc_2D, grids_0, grids_1, N_cc, N_cx):
    """Compute next cc index — depends on (cc, cx) only."""
    for idx_cc in range(N_cc):
        cc = grids_0[idx_cc]
        for idx_cx in range(N_cx):
            cx = grids_1[idx_cx]
            cc_next = cc + cx
            Next_cc_2D[idx_cc, idx_cx] = np.argmin(np.abs(grids_0 - cc_next))


@jit(nopython=True)
def _precompute_next_t_idx(Next_t_2D, grids_3, grids_4, N_T, N_t):
    """Compute next t index — depends on (T, t) only."""
    for idx_T in range(N_T):
        T_val = grids_3[idx_T]
        for idx_t in range(N_t):
            t_val = grids_4[idx_t]
            t_next = t_val + T_val
            Next_t_2D[idx_T, idx_t] = np.argmin(np.abs(grids_4 - t_next))


@jit(nopython=True, parallel=PARALLEL)
def _vi_arrival_fused(V_new, V_old, V_departure,
                      P_survival_4D, Next_cc_2D, Next_t_2D,
                      cu_grid, gamma, cost_sum,
                      N_cc, N_cx, N_cu, N_T, N_t,
                      delta_out):
    """
    Fused kernel for the entire arrival-state VI update.

    Computes V_arrival = max(cu + base_value_4D, gamma * V_departure)
    and the convergence delta in a single pass, with zero temporary arrays.
    """
    V_fail = V_departure[0, 0]
    for idx_cc in prange(N_cc):
        local_delta = np.float32(0.0)
        for idx_cx in range(N_cx):
            for idx_T in range(N_T):
                for idx_t in range(N_t):
                    p = P_survival_4D[idx_cc, idx_cx, idx_T, idx_t]
                    V_surv = V_departure[Next_cc_2D[idx_cc, idx_cx],
                                         Next_t_2D[idx_T, idx_t]]
                    base = (-(np.float32(1.0) - p) * cost_sum
                            + gamma * (p * V_surv
                                       + (np.float32(1.0) - p) * V_fail))
                    q_shut = gamma * V_departure[idx_cc, idx_t]
                    for idx_cu in range(N_cu):
                        q_price = cu_grid[idx_cu] + base
                        v_new = max(q_price, q_shut)
                        V_new[idx_cc, idx_cx, idx_cu, idx_T, idx_t] = v_new
                        d = abs(v_new - V_old[idx_cc, idx_cx, idx_cu, idx_T, idx_t])
                        if d > local_delta:
                            local_delta = d
        delta_out[idx_cc] = local_delta


@jit(nopython=True, parallel=PARALLEL)
def _compute_expected_V_fused(V_arrival, unique_cx, unique_cu, unique_T,
                              weights, N_cc, N_t):
    """
    Compute E[V_arrival | cc, t] via weighted sum over unique customer tuples.
    Replaces the 100K-sample loop with a structured gather.
    """
    K = weights.shape[0]
    expected_V = np.zeros((N_cc, N_t), dtype=np.float32)
    for idx_cc in prange(N_cc):
        for idx_t in range(N_t):
            total = np.float32(0.0)
            for k in range(K):
                total += weights[k] * V_arrival[idx_cc, unique_cx[k],
                                                unique_cu[k], unique_T[k],
                                                idx_t]
            expected_V[idx_cc, idx_t] = total
    return expected_V


# --------------------------------------------------------------------------- #
# ---                         Main Agent Class                            --- #
# --------------------------------------------------------------------------- #

class DiscretizedDPAgent:
    """
    Solves the MDP using value iteration on a discretized state space.
    State: (cum_degradation cc, cust_degradation cx, cust_revenue cu, duration T, active_time t)

    Optimized version — see module docstring for details.
    """

    def __init__(self, N, max_cumulative_context, u_hat, degradation_learner,
                 customer_generator, params):
        self.degradation_learner = degradation_learner
        self.customer_generator = customer_generator
        self.params = params
        self.u_hat = u_hat
        self.gamma = np.float32(params.get('gamma', 0.99))
        self.theta = self.degradation_learner.get_theta()

        self._setup_discretization(N, max_cumulative_context)

        # Value arrays in float32 (halves memory vs float64)
        self.V_arrival = np.zeros(self.grid_shape, dtype=np.float32)
        self.V_departure = np.zeros((self.grid_sizes[0], self.grid_sizes[4]),
                                    dtype=np.float32)

        self.policy_arrival = np.zeros(self.grid_shape, dtype=np.int8)
        self.policy_departure = np.zeros(self.V_departure.shape, dtype=np.int8)

        self._precompute_dynamics()

    # ------------------------------------------------------------------ #
    #  Discretization setup (identical to original)
    # ------------------------------------------------------------------ #

    def _setup_discretization(self, N, max_cumulative_context):
        """Creates the grids for the 5D state space."""
        if isinstance(N, int):
            N = [N, N, N, N, N]

        max_revenue_context = 1.0
        duration_lambda = self.params.get('duration_lambda', 0.1)
        max_rental_duration = -np.log(0.0005) / duration_lambda
        max_customer_degradation_context = 1.0
        max_active_time = self.params.get('max_active_time',
                                          max_rental_duration * 10)

        self.grid_max_vals = [max_cumulative_context,
                              max_customer_degradation_context,
                              max_revenue_context,
                              max_rental_duration,
                              max_active_time]
        self.grid_sizes = N
        self.grid_shape = tuple(N)

        # Store grids as float32 for consistency with value arrays
        self.grids = [
            np.linspace(0, self.grid_max_vals[i], self.grid_sizes[i],
                        dtype=np.float32)
            for i in range(5)
        ]

        logging.info("Discretization setup (optimized):")
        labels = ["Cumulative Context (cc)", "Cust. Degradation (cx)",
                  "Cust. Revenue (cu)", "Rental Duration (T)", "Active Time (t)"]
        for i, label in enumerate(labels):
            logging.info(f"  - {label:30s} {self.grid_sizes[i]} steps "
                        f"up to {self.grid_max_vals[i]:.2f}")

    # ------------------------------------------------------------------ #
    #  State-index lookup (identical interface to original)
    # ------------------------------------------------------------------ #

    def _get_state_indices(self, state_values):
        """Finds the nearest indices for a given continuous state vector."""
        cc, cx, cu, T, t = state_values
        idx_cc = np.argmin(np.abs(self.grids[0] - cc))
        idx_cx = np.argmin(np.abs(self.grids[1] - cx))
        idx_cu = np.argmin(np.abs(self.grids[2] - cu))
        idx_T = np.argmin(np.abs(self.grids[3] - T))
        idx_t = np.argmin(np.abs(self.grids[4] - t))
        return idx_cc, idx_cx, idx_cu, idx_T, idx_t

    # ------------------------------------------------------------------ #
    #  Precomputation  (Steps 1 & 3: reduced dimensionality + unique weights)
    # ------------------------------------------------------------------ #

    def _precompute_dynamics(self, num_samples=100000):
        """Pre-computes dynamics arrays at their natural dimensionality."""
        N_cc, N_cx, N_cu, N_T, N_t = self.grid_sizes
        logging.info(f"Pre-computing dynamics ({N_cc}x{N_cx}x{N_T}x{N_t} = "
                    f"{N_cc*N_cx*N_T*N_t:,} 4D states, "
                    f"vs {N_cc*N_cx*N_cu*N_T*N_t:,} original 5D)...")

        # --- Customer sampling → unique weighted tuples (Step 3) ---
        logging.info(f"  Sampling {num_samples} customers for expectation weights...")
        sampled_customer_indices = []
        for _ in range(num_samples):
            customer = self.customer_generator.generate()
            x, duration = customer['context'], customer['desired_duration']
            cx_val = np.dot(self.theta, x)
            cu_val = np.dot(self.u_hat, x)
            _, idx_cx, idx_cu, idx_T, _ = self._get_state_indices(
                (0, cx_val, cu_val, duration, 0))
            sampled_customer_indices.append((idx_cx, idx_cu, idx_T))

        tuple_counts = Counter(sampled_customer_indices)
        unique_tuples = np.array(list(tuple_counts.keys()), dtype=np.int32)
        weights = np.array(list(tuple_counts.values()), dtype=np.float32)
        weights /= weights.sum()

        self.unique_cx = unique_tuples[:, 0].copy()
        self.unique_cu = unique_tuples[:, 1].copy()
        self.unique_T = unique_tuples[:, 2].copy()
        self.customer_weights = weights
        logging.info(f"  {len(weights)} unique customer tuples "
                    f"(from {num_samples} samples)")

        interarrival_lambda = self.params.get('interarrival_lambda', 1.0)
        self.expected_holding_reward = np.float32(
            -self.params['holding_cost_rate'] * (1.0 / interarrival_lambda))

        # --- Delta_Lambda: 2D (t, T) ---
        logging.info("  Computing Delta_Lambda (2D)...")
        Delta_Lambda = np.zeros((N_t, N_T), dtype=np.float32)
        for idx_t in range(N_t):
            for idx_T in range(N_T):
                t_val = float(self.grids[4][idx_t])
                T_val = float(self.grids[3][idx_T])
                Delta_Lambda[idx_t, idx_T] = np.float32(
                    self.degradation_learner.cum_baseline(t_val + T_val)
                    - self.degradation_learner.cum_baseline(t_val))

        # --- P_survival: 4D (cc, cx, T, t) — Step 1 ---
        logging.info("  Computing P_survival (4D)...")
        self.P_survival_4D = np.zeros((N_cc, N_cx, N_T, N_t), dtype=np.float32)
        _precompute_P_survival_4D(
            self.P_survival_4D,
            self.grids[0], self.grids[1], self.grids[3], self.grids[4],
            N_cc, N_cx, N_T, N_t, Delta_Lambda)

        # --- Next-state indices: 2D each — Step 1 ---
        logging.info("  Computing next-state indices (2D)...")
        self.Next_cc_2D = np.zeros((N_cc, N_cx), dtype=np.int32)
        _precompute_next_cc_idx(
            self.Next_cc_2D, self.grids[0], self.grids[1], N_cc, N_cx)

        self.Next_t_2D = np.zeros((N_T, N_t), dtype=np.int32)
        _precompute_next_t_idx(self.Next_t_2D, self.grids[3], self.grids[4],
                               N_T, N_t)

        # Store cost_sum as float32 for the fused kernel
        self.cost_sum = np.float32(
            self.params['failure_cost'] + self.params['replacement_cost'])

        logging.info("Pre-computation complete.")

    # ------------------------------------------------------------------ #
    #  Value Iteration  (Steps 2-7: all optimizations)
    # ------------------------------------------------------------------ #

    def run_value_iteration(self, num_iterations, tolerance=1e-4):
        """Performs value iteration with fused numba kernel."""
        N_cc, N_cx, N_cu, N_T, N_t = self.grid_sizes
        logging.info(f"\nStarting Value Iteration (optimized) — "
                    f"{N_cc*N_cx*N_cu*N_T*N_t:,} states, "
                    f"float32, fused kernel...")

        history = {'delta': []}

        # Double-buffer for V_arrival (Step 5)
        V_buf = [self.V_arrival,
                 np.zeros(self.grid_shape, dtype=np.float32)]
        buf_idx = 0

        # Pre-allocate per-thread delta array for fused kernel
        delta_arr = np.zeros(N_cc, dtype=np.float32)

        # cu grid for the fused kernel
        cu_grid = self.grids[2]  # shape (N_cu,), float32

        for i in range(num_iterations):
            V_old = V_buf[buf_idx]
            V_new = V_buf[1 - buf_idx]

            # --- 1. Compute E[V_arrival | cc, t] (Step 3) ---
            expected_V_arrival = _compute_expected_V_fused(
                V_old, self.unique_cx, self.unique_cu, self.unique_T,
                self.customer_weights, N_cc, N_t)

            # --- 2. Update Departure Values (2D — cheap) ---
            V_departure_old = self.V_departure.copy()

            q_replace_val = np.float32(
                self.expected_holding_reward
                - self.params['replacement_cost']
                + self.gamma * expected_V_arrival[0, 0])
            q_no_replace = (self.expected_holding_reward
                           + self.gamma * expected_V_arrival)

            np.maximum(q_replace_val, q_no_replace, out=self.V_departure)

            delta_departure = np.max(np.abs(self.V_departure - V_departure_old))

            # --- 3. Update Arrival Values via fused kernel (Steps 2, 5, 7) ---
            _vi_arrival_fused(
                V_new, V_old, self.V_departure,
                self.P_survival_4D, self.Next_cc_2D, self.Next_t_2D,
                cu_grid, self.gamma, self.cost_sum,
                N_cc, N_cx, N_cu, N_T, N_t,
                delta_arr)

            delta_arrival = np.max(delta_arr)
            buf_idx = 1 - buf_idx  # swap buffers

            # --- 4. Convergence check ---
            delta = max(float(delta_arrival), float(delta_departure))
            history['delta'].append(delta)
            if (i + 1) % 10 == 0:
                logging.info(f"Iteration {i+1}/{num_iterations} | "
                           f"Max Delta: {delta:.6f}")

            if delta < tolerance:
                logging.info(f"\nConverged after {i+1} iterations.")
                break

        if i == num_iterations - 1:
            logging.info("\nValue iteration finished (max iterations reached).")

        # Finalize: point V_arrival to the latest buffer
        self.V_arrival = V_buf[buf_idx]

        # --- Compute policies once after convergence (Step 4) ---
        self._compute_final_policies()

        return history

    def _compute_final_policies(self):
        """Extract policies from converged value functions (computed once)."""
        N_cc, N_cx, N_cu, N_T, N_t = self.grid_sizes
        logging.info("Computing final policies...")

        # Departure policy
        expected_V_arrival = _compute_expected_V_fused(
            self.V_arrival, self.unique_cx, self.unique_cu, self.unique_T,
            self.customer_weights, N_cc, N_t)

        q_replace_val = np.float32(
            self.expected_holding_reward
            - self.params['replacement_cost']
            + self.gamma * expected_V_arrival[0, 0])
        q_no_replace = (self.expected_holding_reward
                       + self.gamma * expected_V_arrival)

        self.policy_departure = np.where(
            q_replace_val >= q_no_replace, 2, 3).astype(np.int8)

        # Arrival policy: reconstruct from base_value vs q_shutdown
        # base_value_4D: (N_cc, N_cx, N_T, N_t)
        cc_idx_flat = self.Next_cc_2D.ravel()
        t_idx_flat = self.Next_t_2D.ravel()
        V_next_surv = self.V_departure[
            cc_idx_flat[:, None], t_idx_flat[None, :]
        ].reshape(N_cc, N_cx, N_T, N_t)

        V_fail = self.V_departure[0, 0]
        p = self.P_survival_4D
        base_value = (-(1 - p) * self.cost_sum
                     + self.gamma * (p * V_next_surv + (1 - p) * V_fail))

        # q_give_price[cc,cx,cu,T,t] = cu + base_value[cc,cx,T,t]
        # q_shutdown[cc,t] = gamma * V_departure[cc,t]
        cu_5d = self.grids[2][np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        base_5d = base_value[:, :, np.newaxis, :, :]
        q_shutdown_5d = (self.gamma
                        * self.V_departure[:, np.newaxis, np.newaxis,
                                           np.newaxis, :])

        self.policy_arrival = np.where(
            cu_5d + base_5d >= q_shutdown_5d, 0, 1).astype(np.int8)

        logging.info("Policies computed.")

    # ------------------------------------------------------------------ #
    #  Policy interface (identical to original)
    # ------------------------------------------------------------------ #

    def get_policy(self, type):
        """Returns a function that represents the learned greedy policy."""
        def greedy_policy_fn(state, policy_kwargs):
            cc, cx, cu, T, t, phase = state
            if phase == 0:  # Arrival
                indices = self._get_state_indices((cc, cx, cu, T, t))
                return int(self.policy_arrival[indices])
            else:  # Departure
                idx_cc, _, _, _, idx_t = self._get_state_indices(
                    (cc, 0, 0, 0, t))
                return int(self.policy_departure[idx_cc, idx_t])

        def epsilon_greedy_policy_fn(state, kwargs={'current_epsilon': 0.1}):
            epsilon = kwargs.get('current_epsilon', 0.1)
            if np.random.rand() < epsilon:
                return np.random.choice([0, 1]) if state[5] == 0 \
                    else np.random.choice([2, 3])
            else:
                return greedy_policy_fn(state, kwargs)

        def decaying_epsilon_greedy_fn(state,
                                       kwargs={'current_epsilon': 0.1,
                                               'decay_rate': 0.99,
                                               'step': 0}):
            initial_epsilon = kwargs.get('current_epsilon', 0.1)
            min_epsilon = 0.001
            current_epsilon = max(
                min_epsilon,
                initial_epsilon * (kwargs['decay_rate'] ** kwargs['step']))
            if np.random.rand() < current_epsilon:
                return np.random.choice([0, 1]) if state[5] == 0 \
                    else np.random.choice([2, 3])
            else:
                return greedy_policy_fn(state, kwargs)

        if type == 'greedy':
            return greedy_policy_fn
        elif type == 'epsilon_greedy':
            return epsilon_greedy_policy_fn
        elif type == 'decaying_epsilon_greedy':
            return decaying_epsilon_greedy_fn

    # ------------------------------------------------------------------ #
    #  Save / Load (compatible interface)
    # ------------------------------------------------------------------ #

    def save_policy(self, filepath):
        """Saves the essential policy components to a file."""
        policy_data = {
            'grids': self.grids,
            'policy_arrival': self.policy_arrival,
            'policy_departure': self.policy_departure,
            'V_arrival': self.V_arrival,
            'V_departure': self.V_departure,
            'params': self.params,
            'u_hat': self.u_hat,
            'degradation_learner': self.degradation_learner,
            'customer_generator': self.customer_generator,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
        logging.info(f"Policy saved to {filepath}")

    @staticmethod
    def load_policy(filepath):
        """Loads the policy components from a file."""
        with open(filepath, 'rb') as f:
            policy_data = pickle.load(f)

        agent = DiscretizedDPAgent(
            N=[len(g) for g in policy_data['grids']],
            max_cumulative_context=float(policy_data['grids'][0][-1]),
            u_hat=policy_data['u_hat'],
            degradation_learner=policy_data['degradation_learner'],
            customer_generator=policy_data['customer_generator'],
            params=policy_data['params'],
        )
        agent.policy_arrival = policy_data['policy_arrival']
        agent.policy_departure = policy_data['policy_departure']
        agent.V_arrival = policy_data['V_arrival']
        agent.V_departure = policy_data['V_departure']
        return agent
