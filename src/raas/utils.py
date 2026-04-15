import numpy as np
import pandas as pd
import raas.config as config
from raas.hazard_models import ExponentialHazard
from raas.optimized_discrete_policy import DiscretizedDPAgent
from raas.simulation import CustomerGenerator

def correct_signs(u, c, d):
    """
    Args:
        u: (d,) array of agent utilities
        c: (d,) array of context
        d: int dimension of the context

    Returns:
        Flip signs of c until u @ c > 0
    """

    while u @ c < 0:
        c = -c
    return c

# def calculate_rolling_rate(df, time_col, value_col, window_size):
#     # 1. Sort and prepare
#     df = df.sort_values(time_col).reset_index(drop=True)
#     times = df[time_col].values
#     values = df[value_col].values
    
#     # 2. Find start indices for each window
#     # window = [t_i - window_size, t_i]
#     start_times = times - window_size
#     start_indices = np.searchsorted(times, start_times, side='left')
    
#     # 3. Cumulative sum with a leading zero for easy slicing
#     # cumsum[i+1] contains the sum of values[0] through values[i]
#     cumsum = np.zeros(len(values) + 1)
#     np.cumsum(values, out=cumsum[1:])
    
#     # 4. Calculate sums in windows
#     # Window sum for values[start_idx : i+1] is cumsum[i+1] - cumsum[start_idx]
#     current_indices = np.arange(len(values)) + 1
#     window_sums = cumsum[current_indices] - cumsum[start_indices]
    
#     # 5. Calculate the denominator (the effective time elapsed)
#     # For t < window_size, the duration is just (current_time - start_of_data)
#     # For t >= window_size, the duration is window_size
#     time_since_start = times - times[0]
#     durations = np.minimum(window_size, time_since_start)
    
#     # Handle the very first point where duration would be 0
#     # If duration is 0, the rate is just the value itself (instantaneous)
#     # or we can treat the duration as a tiny epsilon.
#     durations = np.where(durations == 0, 1e-9, durations)
    
#     profit_rate = window_sums / durations
    
#     return pd.Series(profit_rate, index=df.index)

def calculate_rolling_rate_ewma(df, time_col, value_col, window_size):
    """
    Calculates an Exponentially Weighted Moving Average rate for irregular time series.
    'window_size' here acts as the decay time constant (tau).
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    times = df[time_col].values
    values = df[value_col].values
    
    n = len(values)
    ewma_rate = np.zeros(n)
    
    # Initialize at the first data point
    ewma_rate[0] = values[0] 
    
    # We use a loop here because EWMA is inherently recursive.
    # For massive datasets, this can be accelerated with Numba.
    for i in range(1, n):
        delta_t = times[i] - times[i-1]
        
        # Alpha determines how much of the 'new' value to keep
        # based on how much time has passed.
        alpha = 1 - np.exp(-delta_t / window_size)
        
        # Update the rate: (Current Value) + (Decayed previous average)
        # Note: To keep it as a 'rate', we scale the input value 
        # or treat the EWMA as the smoothed state of the profit flow.
        ewma_rate[i] = alpha * values[i] + (1 - alpha) * ewma_rate[i-1]
        
    return pd.Series(ewma_rate, index=df.index)

def calculate_rolling_rate(df, time_col, value_col, window_size):
    df = df.sort_values(time_col).reset_index(drop=True)
    times = df[time_col].values
    values = df[value_col].values
    
    half_w = window_size / 2
    
    # Define the centered window boundaries
    # t=30, size=100 -> [0, 60]
    # t=100, size=100 -> [50, 150]
    window_starts = np.maximum(0, times - half_w)
    window_ends = np.minimum(times[-1], times + half_w)
    
    # Use searchsorted to find indices for these time boundaries
    start_indices = np.searchsorted(times, window_starts, side='left')
    end_indices = np.searchsorted(times, window_ends, side='right') - 1
    
    # Cumulative sum for O(1) window sum calculation
    cumsum = np.zeros(len(values) + 1)
    np.cumsum(values, out=cumsum[1:])
    
    # Calculate sum of values in [start_idx, end_idx]
    # sum = cumsum[end_idx + 1] - cumsum[start_idx]
    window_sums = cumsum[end_indices + 1] - cumsum[start_indices]
    
    # The actual duration of the window used
    actual_durations = window_ends - window_starts
    
    # Avoid division by zero at t=0
    actual_durations = np.where(actual_durations == 0, 1e-9, actual_durations)
    
    profit_rate = window_sums / actual_durations
    
    return pd.Series(profit_rate, index=df.index)

class PerfectDegradationLearner:
    def __init__(self, d, theta_true, hazard_model):
        self.d = d
        self.theta_true = theta_true
        self.hazard_model = hazard_model  # Placeholder, not used
        
    def get_theta(self):
        return self.theta_true
    
    def cum_baseline(self, t):
        return self.hazard_model.Lambda_0(t)
    
    def inverse_cum_baseline(self, u):
        return self.hazard_model.Lambda_0_inverse(u)

def get_perfect_degradation_learner(
    sample_size=150000, 
    iterations=200, 
    mdp_params=config.mdp_params, 
    baseline_hazard_lambda=config.LAMBDA_VAL
    ):
    usage_exp_hazard_model = ExponentialHazard(lambda_val=baseline_hazard_lambda)
    perfect_degradation_learner = PerfectDegradationLearner(
        d=config.D, 
        theta_true=config.THETA_TRUE,
        hazard_model=usage_exp_hazard_model,
    )
    customer_gen = CustomerGenerator(
        d=config.D,
        context_sampler=config.context_sampler,
        rental_sampler=config.rental_sampler,
        interarrival_sampler=config.interarrival_sampler
    )
    
    perfect_dpagent = DiscretizedDPAgent(
        N=config.training_hyperparams['N'],
        max_cumulative_context=config.training_hyperparams['max_cumulative_context'],
        u_hat=config.UTILITY_TRUE,
        degradation_learner=perfect_degradation_learner,
        customer_generator=customer_gen,
        params=mdp_params,
        num_samples_precompute=sample_size,
    )
    
    perfect_dpagent.run_value_iteration(iterations)
    perfect_policy = perfect_dpagent.get_policy('greedy')
    
    return perfect_degradation_learner, perfect_dpagent, perfect_policy