import numpy as np
from tqdm import tqdm
import pickle
from numba import jit, prange
import logging

parallel = True # issue with parallel on Apple Silicon

@jit(nopython=True)
def _get_state_indices_numba(cc, cx, cu, T, t, grids_0, grids_1, grids_2, grids_3, grids_4):
    """Numba-jitted version to find the nearest indices in the grids."""
    idx_cc = np.argmin(np.abs(grids_0 - cc))
    idx_cx = np.argmin(np.abs(grids_1 - cx))
    idx_cu = np.argmin(np.abs(grids_2 - cu))
    idx_T = np.argmin(np.abs(grids_3 - T))
    idx_t = np.argmin(np.abs(grids_4 - t))
    return idx_cc, idx_cx, idx_cu, idx_T, idx_t

@jit(nopython=True, parallel=parallel)
def _precompute_arrival_dynamics_numba(
    R_arrival, P_survival, Next_cc_idx, Next_t_idx,
    grids_tuple, grid_sizes, Delta_Lambda, failure_cost, replacement_cost
):
    """Fills the dynamics arrays for the arrival state using the state space."""
    N_cc, N_cx, N_cu, N_T, N_t = grid_sizes
    grids_0, grids_1, grids_2, grids_3, grids_4 = grids_tuple

    for idx_cc in prange(N_cc):
        for idx_cx in range(N_cx):
            for idx_cu in range(N_cu):
                for idx_T in range(N_T):
                    for idx_t in range(N_t):
                        cc, cx, cu, T, t = grids_0[idx_cc], grids_1[idx_cx], grids_2[idx_cu], grids_3[idx_T], grids_4[idx_t]

                        # Revenue 'cu' is now directly part of the state
                        revenue = cu
                        
                        # Use precomputed Delta_Lambda from estimated cum_baseline
                        incremental_hazard = Delta_Lambda[idx_t, idx_T] * np.exp(cc + cx)
                        
                        p_survival = np.exp(-incremental_hazard)
                        P_survival[idx_cc, idx_cx, idx_cu, idx_T, idx_t] = p_survival

                        R_surv = revenue
                        R_fail = revenue - failure_cost - replacement_cost
                        R_arrival[idx_cc, idx_cx, idx_cu, idx_T, idx_t] = p_survival * R_surv + (1 - p_survival) * R_fail

                        # Next state upon survival (cc + cx, t + T)
                        cc_next_surv = cc + cx
                        t_next_surv = t + T
                        idx_cc_next, _, _, _, idx_t_next = _get_state_indices_numba(
                            cc_next_surv, 0.0, 0.0, 0.0, t_next_surv,
                            grids_0, grids_1, grids_2, grids_3, grids_4
                        )
                        Next_cc_idx[idx_cc, idx_cx, idx_cu, idx_T, idx_t] = idx_cc_next
                        Next_t_idx[idx_cc, idx_cx, idx_cu, idx_T, idx_t] = idx_t_next

@jit(nopython=True, parallel=parallel)
def _compute_expected_V_arrival_numba(V_arrival, sampled_customer_indices_arr, N_cc, N_t):
    """Computes the expected value of V_arrival for a given (cc, t), averaged over all customers."""
    num_samples = sampled_customer_indices_arr.shape[0]
    expected_V = np.zeros((N_cc, N_t))
    
    for idx_cc in prange(N_cc):
        for idx_t in range(N_t):
            total_value = 0.0
            for k in range(num_samples):
                idx_cx, idx_cu, idx_T = sampled_customer_indices_arr[k, 0], sampled_customer_indices_arr[k, 1], sampled_customer_indices_arr[k, 2]
                total_value += V_arrival[idx_cc, idx_cx, idx_cu, idx_T, idx_t]
            expected_V[idx_cc, idx_t] = total_value / num_samples
            
    return expected_V

# --------------------------------------------------------------------------- #
# ---                         Main Agent Class                            --- #
# --------------------------------------------------------------------------- #

class DiscretizedDPAgent:
    """
    Solves the MDP using value iteration on a discretized state space.
    State: (cum_degradation cc, cust_degradation cx, cust_revenue cu, duration T, active_time t)
    """
    def __init__(self, N, max_cumulative_context, u_hat, degradation_learner, customer_generator, params):
        self.degradation_learner = degradation_learner
        self.customer_generator = customer_generator
        self.params = params
        self.u_hat = u_hat
        self.gamma = params.get('gamma', 0.99)
        self.theta = self.degradation_learner.get_theta()
        
        self._setup_discretization(N, max_cumulative_context)

        # V_arrival is 5D, V_departure is 2D (cc, t)
        self.V_arrival = np.zeros(self.grid_shape)
        self.V_departure = np.zeros((self.grid_sizes[0], self.grid_sizes[4]))
        
        self.policy_arrival = np.zeros(self.grid_shape, dtype=int)
        self.policy_departure = np.zeros(self.V_departure.shape, dtype=int)

        self._precompute_dynamics()

    def _setup_discretization(self, N, max_cumulative_context, num_samples_for_max_val=1000):
        """Creates the grids for the new 5D state space."""
        if isinstance(N, int):
            N = [N, N, N, N, N]

        max_revenue_context = 1.0
        duration_lambda = self.params.get('duration_lambda', 0.1)
        max_rental_duration = -np.log(0.0005) / duration_lambda  # Corrected to / for Exp quantile
        max_customer_degradation_context = 1.0
        max_active_time = self.params.get('max_active_time', max_rental_duration * 10)  # Added for t

        self.grid_max_vals = [max_cumulative_context, max_customer_degradation_context, max_revenue_context, max_rental_duration, max_active_time]
        self.grid_sizes = N
        self.grid_shape = tuple(N)
        
        self.grids = [
            np.linspace(0, self.grid_max_vals[0], self.grid_sizes[0]),
            np.linspace(0, self.grid_max_vals[1], self.grid_sizes[1]),
            np.linspace(0, self.grid_max_vals[2], self.grid_sizes[2]),
            np.linspace(0, self.grid_max_vals[3], self.grid_sizes[3]),
            np.linspace(0, self.grid_max_vals[4], self.grid_sizes[4])  # Added grid for t
        ]
        
        logging.info("Discretization setup:")
        logging.info(f"  - Cumulative Context (cc):    {self.grid_sizes[0]} steps up to {self.grid_max_vals[0]:.2f}")
        logging.info(f"  - Cust. Degradation (cx):     {self.grid_sizes[1]} steps up to {self.grid_max_vals[1]:.2f}")
        logging.info(f"  - Cust. Revenue (cu):         {self.grid_sizes[2]} steps up to {self.grid_max_vals[2]:.2f}")
        logging.info(f"  - Rental Duration (T):        {self.grid_sizes[3]} steps up to {self.grid_max_vals[3]:.2f} (99.95th percentile)")
        logging.info(f"  - Active Time (t):            {self.grid_sizes[4]} steps up to {self.grid_max_vals[4]:.2f}")

    def _get_state_indices(self, state_values):
        """Finds the nearest indices for a given continuous state vector."""
        cc, cx, cu, T, t = state_values
        idx_cc = np.argmin(np.abs(self.grids[0] - cc))
        idx_cx = np.argmin(np.abs(self.grids[1] - cx))
        idx_cu = np.argmin(np.abs(self.grids[2] - cu))
        idx_T = np.argmin(np.abs(self.grids[3] - T))
        idx_t = np.argmin(np.abs(self.grids[4] - t))
        return idx_cc, idx_cx, idx_cu, idx_T, idx_t
    
    def _precompute_dynamics(self, num_samples=100000):
        """Pre-computes dynamics for the state space."""
        logging.info(f"Pre-computing expectations from {num_samples} customer samples...")
        
        sampled_customer_indices = []
        for _ in range(num_samples):
            customer = self.customer_generator.generate()
            x, duration = customer['context'], customer['desired_duration']
            cx_val = np.dot(self.theta, x)
            cu_val = np.dot(self.u_hat, x)
            _, idx_cx, idx_cu, idx_T, _ = self._get_state_indices((0, cx_val, cu_val, duration, 0))  # CHANGED: Added t=0
            sampled_customer_indices.append((idx_cx, idx_cu, idx_T))
        
        self.sampled_customer_indices_arr = np.array(sampled_customer_indices, dtype=np.int32)
        
        interarrival_lambda = self.params.get('interarrival_lambda', 1.0)
        self.expected_holding_reward = -self.params['holding_cost_rate'] * (1.0 / interarrival_lambda)
        
        self.R_arrival = np.zeros(self.grid_shape)  # Renamed from R_arrival_expected
        self.P_survival = np.zeros(self.grid_shape)
        self.Next_cc_idx = np.zeros(self.grid_shape, dtype=np.int32)  # Separate for 2D departure
        self.Next_t_idx = np.zeros(self.grid_shape, dtype=np.int32) 
        
        # Precompute Delta_Lambda from estimated cum_baseline
        N_t, N_T = self.grid_sizes[4], self.grid_sizes[3]
        Delta_Lambda = np.zeros((N_t, N_T))
        for idx_t in range(N_t):
            for idx_T in range(N_T):
                t_val = self.grids[4][idx_t]
                T_val = self.grids[3][idx_T]
                Delta_Lambda[idx_t, idx_T] = self.degradation_learner.cum_baseline(t_val + T_val) - self.degradation_learner.cum_baseline(t_val)
        
        logging.info("Starting Numba-accelerated pre-computation of arrival dynamics...")
        _precompute_arrival_dynamics_numba(
            self.R_arrival, self.P_survival, self.Next_cc_idx, self.Next_t_idx,
            tuple(self.grids), self.grid_shape,
            Delta_Lambda,
            self.params['failure_cost'], self.params['replacement_cost'],
        )
        logging.info("Pre-computation complete. ✅")

    def run_value_iteration(self, num_iterations, tolerance=1e-4):
        """Performs value iteration on the state space."""
        logging.info("\nStarting Value Iteration...")
        history = {'delta': []}
        for i in range(num_iterations):
            # --- 1. Update Departure State Values (2D array) --- 
            V_departure_old = self.V_departure.copy()
            
            # Expected value of starting in an arrival state, averaged over all customers
            expected_V_arrival = _compute_expected_V_arrival_numba(
                self.V_arrival, self.sampled_customer_indices_arr, self.grid_sizes[0], self.grid_sizes[4]  
            )
            
            # Q-value for action 'replace' (2): cc,t reset to 0,0.
            q_replace_value = self.expected_holding_reward - self.params['replacement_cost'] + self.gamma * expected_V_arrival[0, 0]
            q_replace = np.full(self.V_departure.shape, q_replace_value)
            
            # Q-value for action 'no_replace' (3): (cc,t) preserved.
            q_no_replace = self.expected_holding_reward + self.gamma * expected_V_arrival
            
            self.V_departure = np.maximum(q_replace, q_no_replace)
            self.policy_departure = np.argmax(np.stack([q_replace, q_no_replace], axis=-1), axis=-1) + 2

            # --- 2. Update Arrival State Values (5D array) --- 
            V_arrival_old = self.V_arrival.copy()
            
            # Q-value for action 'shutdown' (1): transitions to a departure state with same (cc, t).
            # Reshape to broadcast correctly with the 5D V_arrival.
            V_next_shutdown = self.V_departure.reshape(self.grid_sizes[0], 1, 1, 1, self.grid_sizes[4])
            q_shutdown = 0 + self.gamma * V_next_shutdown

            # Q-value for action 'give_price' (0)
            V_next_survival = self.V_departure[self.Next_cc_idx, self.Next_t_idx]  # Use separate idx
            V_next_failure = self.V_departure[0, 0]  # Failure always resets to (cc=0, t=0)
            
            expected_V_next_give_price = self.P_survival * V_next_survival + (1 - self.P_survival) * V_next_failure
            q_give_price = self.R_arrival + self.gamma * expected_V_next_give_price

            self.V_arrival = np.maximum(q_give_price, q_shutdown)
            
            # For stacking, broadcast q_shutdown to match q_give_price's shape
            q_shutdown_broadcasted = np.broadcast_to(q_shutdown, self.V_arrival.shape)
            self.policy_arrival = np.argmax(np.stack([q_give_price, q_shutdown_broadcasted], axis=-1), axis=-1)

            # --- 3. Check for Convergence ---
            delta = max(
                np.max(np.abs(self.V_arrival - V_arrival_old)),
                np.max(np.abs(self.V_departure - V_departure_old))
            )
            history['delta'].append(delta)
            if (i + 1) % 10 == 0:
                logging.info(f"Iteration {i+1}/{num_iterations} | Max Change (Delta): {delta:.6f}")
            
            if delta < tolerance:
                logging.info(f"\nValue iteration converged after {i+1} iterations. ✨")
                break
        
        if i == num_iterations - 1:
            logging.info("\nValue iteration finished (max iterations reached).")
        return history

    def get_policy(self, type):
        """Returns a function that represents the learned greedy policy."""
        def greedy_policy_fn(state, policy_kwargs):
            """
            Args: state (tuple): (cc, cx, cu, T, t, phase). # CHANGED: Added t
            """
            cc, cx, cu, T, t, phase = state
            
            if phase == 0:  # Arrival
                indices = self._get_state_indices((cc, cx, cu, T, t))
                return self.policy_arrival[indices]
            else:  # Departure
                idx_cc, _, _, _, idx_t = self._get_state_indices((cc, 0, 0, 0, t))
                return self.policy_departure[idx_cc, idx_t]

        def epsilon_greedy_policy_fn(state, kwargs={'current_epsilon': 0.1}):
            epsilon = kwargs.get('current_epsilon', 0.1)
            if np.random.rand() < epsilon:
                return np.random.choice([0, 1]) if state[5] == 0 else np.random.choice([2, 3])
            else:
                return greedy_policy_fn(state, kwargs)

        def decaying_epsilon_greedy_fn(state, kwargs={'current_epsilon': 0.1,'decay_rate': 0.99, 'step': 0}):
            initial_epsilon = kwargs.get('current_epsilon', 0.1)
            min_epsilon = 0.001
            current_epsilon = max(min_epsilon, initial_epsilon * (kwargs['decay_rate'] ** kwargs['step']))
            if np.random.rand() < current_epsilon:
                return np.random.choice([0, 1]) if state[5] == 0 else np.random.choice([2, 3])
            else:
                return greedy_policy_fn(state, kwargs)

        if type == 'greedy':
            policy_fn = greedy_policy_fn
        elif type == 'epsilon_greedy':
            policy_fn = epsilon_greedy_policy_fn
        elif type == 'decaying_epsilon_greedy':
            policy_fn = decaying_epsilon_greedy_fn
        return policy_fn

    def save_policy(self, filepath):
        """Saves the essential policy components to a file."""
        policy_data = {
            'grids': self.grids,  # CHANGED: Now 5 grids
            'policy_arrival': self.policy_arrival,
            'policy_departure': self.policy_departure,
            'V_arrival': self.V_arrival,
            'V_departure': self.V_departure,
            'params': self.params,
            'u_hat': self.u_hat,
            'degradation_learner': self.degradation_learner,
            'customer_generator': self.customer_generator
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
            max_cumulative_context=policy_data['grids'][0][-1],
            u_hat=policy_data['u_hat'],  # Placeholder, not used in loaded policy
            degradation_learner=policy_data['degradation_learner'],  # Placeholder, not used in loaded policy
            customer_generator=policy_data['customer_generator'],  # Placeholder, not used in loaded policy
            params=policy_data['params']
        )
        agent.policy_arrival = policy_data['policy_arrival']
        agent.policy_departure = policy_data['policy_departure']
        agent.V_arrival = policy_data['V_arrival']
        agent.V_departure = policy_data['V_departure']
        return agent