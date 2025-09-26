import numpy as np
from utils import unit_ball_sample

def context_sampler() -> np.ndarray:
    """Samples a customer's context vector uniformly from the unit ball."""
    return np.abs(unit_ball_sample(D))

def rental_sampler() -> float:
    """Samples a customer's desired rental duration from an exponential distribution."""
    return np.random.exponential(scale=10.0)

def interarrival_sampler() -> float:
    """Samples the time until the next customer arrives."""
    return np.random.exponential(scale=5.0)

# --- 1. Simulation Configuration ---
D = 4                                  # Dimension of context vectors
LAMBDA_VAL = 0.001                     # Baseline hazard constant
NUM_CUSTOMERS = 40000                   # Total number of customers to simulate, i.e. T

# Ground truth vectors
THETA_TRUE = np.array([0.5, 0.2, 0.4, 0.3])#$, 0.4])    # For degradation
UTILITY_TRUE = np.array([0.372450167, 0.10850869, 0.33930126, 0.71356037])

# --- Machine's Pricing Vector 'r' ---
# This is a fallback pricing vector, when we don't feed u_hat to calculate_price
PRICING_R = np.zeros(D)

centroid_params = {
    # 'num_samples': 2000,
    # 'thin': None,
    # 'burn_in': 500 * D ** 2,
    # 'tol': 1e-4,
    # 'rho_target': 0.01
}

termination_rule = lambda diameter: diameter < 0.0005  # Example custom termination rule


mdp_params = {
    'duration_lambda': 10.0,
    'interarrival_lambda': 5.0,
    'replacement_cost': 1.5,   # Cost to replace the machine
    'failure_cost': 0.75,      # Additional penalty for in-service failure
    'holding_cost_rate': 0.02,   # Cost per unit of idle time
    'gamma': 0.99,             # Discount factor
    'learning_rate': 1e-3,      # Learning rate for the Adam optimizer
    'target_update_freq': 10    # How often to update the target network (in iterations)
}


training_hyperparams = {
    # For FQI
    'num_iterations': 1, # Number of training iterations per policy update
    'dataset_size': 50000,      # Number of transitions to generate for the offline dataset
    'batch_size': 256,           # Batch size for training

    # For discrete DP
    'N': [50, 50, 50, 50, 50], # grid sizes [cum_context, context, revenue, duration, active_time]
    'max_cumulative_context': 8.0,
    # 'max_active_time': 150.0,
    'num_value_iterations': 100,
    
}

incentive_constant = 1.1

policy_type = 'decaying_epsilon_greedy'
policy_kwargs = {
    'current_epsilon': 0.10,
    'decay_rate': 0.95,
    'step': 0,
}


