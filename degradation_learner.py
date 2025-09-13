import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import pandas as pd

def cox_partial_loglik(theta, long_df):
    long_df = long_df.sort_values('stop').reset_index(drop=True)
    event_times = long_df[long_df['event'] == 1]['stop'].unique()
    logL = 0.0
    x_cols = [col for col in long_df.columns if col.startswith('X')]
    X = long_df[x_cols].values  # n x d matrix
    
    for t in event_times:
        # Events at t (for ties)
        events_at_t = long_df[(long_df['stop'] == t) & (long_df['event'] == 1)]
        # Risk set: intervals covering t
        risk_set = long_df[(long_df['start'] < t) & (long_df['stop'] >= t)]
        if risk_set.empty:
            continue
        
        # Denominator: sum exp(theta @ X_j) over risk
        exp_terms_risk = np.exp(np.dot(X[risk_set.index], theta))
        denom = np.sum(exp_terms_risk)
        
        # For Breslow (simple ties handling): sum over events (theta @ X_k) - num_events * log(denom)
        num_events = len(events_at_t)
        for idx in events_at_t.index:
            logL += np.dot(X[idx], theta)
        logL -= num_events * np.log(denom)
    
    return logL

# To optimize: minimize -logL
def neg_loglik(theta, long_df):
    return -cox_partial_loglik(theta, long_df)

# Analytic gradient (for faster optim)
def cox_partial_gradient(theta, long_df):
    long_df = long_df.sort_values('stop').reset_index(drop=True)
    event_times = long_df[long_df['event'] == 1]['stop'].unique()
    grad = np.zeros_like(theta)  # d=5
    x_cols = [col for col in long_df.columns if col.startswith('X')]
    X = long_df[x_cols].values
    
    for t in event_times:
        events_at_t = long_df[(long_df['stop'] == t) & (long_df['event'] == 1)]
        risk_set = long_df[(long_df['start'] < t) & (long_df['stop'] >= t)]
        exp_terms_risk = np.exp(np.dot(X[risk_set.index], theta))
        denom = np.sum(exp_terms_risk)
        weighted_X_risk = np.dot(exp_terms_risk, X[risk_set.index]) / denom
        
        num_events = len(events_at_t)
        for idx in events_at_t.index:
            grad += X[idx]
        grad -= num_events * weighted_X_risk
    
    return grad

def neg_gradient(theta, long_df):
    return -cox_partial_gradient(theta, long_df)
    
def breslow_baseline_estimator(long_df, theta_hat):
    """
    Computes Breslow cumulative baseline hazard Lambda_0(t) and approximate discrete lambda_0(t).
    
    Args:
        long_df (pd.DataFrame): Start-stop format with 'start', 'stop', 'event', and X0-X4 columns.
        theta_hat (np.array): Estimated theta (d=5).
    
    Returns:
        pd.DataFrame: With 'time' (event times), 'Lambda_0' (cumulative), 'lambda_0' (discrete hazard).
    """
    long_df = long_df.sort_values('stop').reset_index(drop=True)
    event_times = sorted(long_df[long_df['event'] == 1]['stop'].unique())
    Lambda_0 = np.zeros(len(event_times))
    cum_Lambda = 0.0
    x_cols = [col for col in long_df.columns if col.startswith('X')]
    X = long_df[x_cols].values  # n x d
    
    for i, t in enumerate(event_times):
        events_at_t = long_df[(long_df['stop'] == t) & (long_df['event'] == 1)]
        risk_set = long_df[(long_df['start'] < t) & (long_df['stop'] >= t)]
        if risk_set.empty:
            continue
        exp_terms_risk = np.exp(np.dot(X[risk_set.index], theta_hat))
        denom = np.sum(exp_terms_risk)
        d_k = len(events_at_t)
        increment = d_k / denom if denom > 0 else 0
        cum_Lambda += increment
        Lambda_0[i] = cum_Lambda
    
    breslow_df = pd.DataFrame({'time': event_times, 'Lambda_0': Lambda_0})
    
    breslow_df['delta_Lambda'] = breslow_df['Lambda_0'].diff().fillna(breslow_df['Lambda_0'].iloc[0])
    breslow_df['delta_t'] = breslow_df['time'].diff().fillna(breslow_df['time'].iloc[0])
    breslow_df['lambda_0'] = breslow_df['delta_Lambda'] / breslow_df['delta_t']

    return breslow_df

class DegradationLearner:
    def __init__(self, d, initial_theta=None):
        self.d = d
        self.initial_theta = np.zeros(d) if initial_theta is None else initial_theta

    def fit(self, data):
        data['life_id'] = (data['event'].shift(1).fillna(-99) == 1).cumsum()  # 0 after breakdown

        x_cols = [f'X{j}' for j in range(self.d)]
        data = pd.concat([data[['life_id', 'start', 'stop', 'event']+x_cols]], axis=1)

        self.fit_usage_hazard(data)
        self.fit_baseline_hazard(data)
        return None

    def fit_usage_hazard(self, data):
        bounds = [(0, None)] * self.d
        res = minimize(
            neg_loglik, 
            self.initial_theta, 
            args=(data,), 
            jac=neg_gradient, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'disp': True, 'maxiter': 1000}
        )
        self.theta = res.x
        return None
    
    def fit_baseline_hazard(self, data):
        breslow_df = breslow_baseline_estimator(
            data, 
            self.get_theta()
        )
        breslow_df = breslow_df[breslow_df['delta_t'] > 0]
        times = breslow_df['time'].values
        lambda_step = breslow_df['lambda_0'].values

        # KDE for smoothing
        kde = gaussian_kde(times, bw_method='silverman', weights=lambda_step)
        self.kde = kde

    def get_theta(self):
        return self.theta
    
    def predict_failure_prob(self, sum_before, current_context, duration, calendar_time=None):
        """
        Predicts failure probability for a new rental.
        
        Args:
            sum_before (np.array): Cumulative context before this rental (shape: (d,)).
            current_context (np.array): Current renter's context (shape: (d,)).
            duration (float): Requested rental duration T.
            calendar_time (float): Optional calendar time (not used here).
        
        Returns:
            float: Estimated P(failure during [0, T] | X_total).
        """
        if not hasattr(self, 'theta') or not hasattr(self, 'kde'):
            raise ValueError("Model must be fitted first.")
        
        X_total = sum_before + current_context
        exp_term = np.exp(np.dot(self.theta, X_total))
        
        # Integrate smoothed lambda_0(u) from 0 to duration
        def lambda_0_integrand(u):
            return self.kde(u)[0]  # KDE evaluate returns array
        
        Lambda_0_T, _ = quad(lambda_0_integrand, 0, duration, limit=100, epsabs=1e-8)  # Numerical integration
        
        survival_prob = np.exp(-Lambda_0_T * exp_term)
        failure_prob = 1 - survival_prob
        
        return failure_prob