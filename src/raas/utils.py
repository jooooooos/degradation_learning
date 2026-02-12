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

def calculate_rolling_rate(df, time_col, value_col, window_size):
    """
    Calculates the rate of a value over a rolling time window on irregular time series data.

    Args:
        df (pd.DataFrame): The input dataframe.
        time_col (str): The name of the column with time data.
        value_col (str): The name of the column with values to aggregate (e.g., 'net_profit').
        window_size (int): The duration of the rolling time window.

    Returns:
        pd.Series: A series containing the calculated rolling rate for each row.
    """
    # Ensure the dataframe is sorted by time, which is crucial.
    df = df.sort_values(time_col).reset_index(drop=True)
    
    times = df[time_col].values
    values = df[value_col].values
    
    # For each end time `t_i`, find the start time `t_i - window`.
    start_times = times - window_size
    
    # Use searchsorted to find the index where each start_time would be inserted.
    # This gives us the starting index of each time window efficiently.
    start_indices = np.searchsorted(times, start_times, side='left')
    
    # Use a cumulative sum to efficiently calculate the sum over any slice [j, i].
    value_cumsum = np.cumsum(values)
    
    # The sum for a window ending at `i` is cumsum[i] - cumsum[start_index - 1].
    # We create a shifted cumulative sum array to handle the `start_index - 1` lookup.
    shifted_cumsum = np.concatenate(([0], value_cumsum[:-1]))
    
    # Calculate the sum of values within each rolling window.
    window_sums = value_cumsum - shifted_cumsum[start_indices]
    
    # The rate is the sum of profit in the window divided by the window's duration.
    profit_rate = window_sums / window_size
    
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
    )
    
    perfect_dpagent._precompute_dynamics(sample_size)
    perfect_dpagent.run_value_iteration(iterations)
    perfect_policy = perfect_dpagent.get_policy('greedy')
    
    return perfect_degradation_learner, perfect_dpagent, perfect_policy