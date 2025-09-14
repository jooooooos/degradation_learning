import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

from policy import DPAgent
from simulation import Simulator, CustomerGenerator
from hazard_models import ExponentialHazard
from utility_learner import ProjectedVolumeLearner, diam
from degradation_learner import DegradationLearner

from utils import unit_ball_rejection_sample, correct_signs
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import logging
logging.basicConfig(level=logging.INFO)

np.set_printoptions(suppress=True)

# --- 1. Simulation Configuration ---
D = 4                                  # Dimension of context vectors
LAMBDA_VAL = 0.001                     # Baseline hazard constant
NUM_CUSTOMERS = 2000                   # Total number of customers to simulate, i.e. T

# Set a random seed for reproducibility
np.random.seed(41)

# Ground truth vectors
THETA_TRUE = np.array([0.5, 0.2, 0.1, 0.3, 0.4])    # For degradation
UTILITY_TRUE = context_sampler()  # For customer's willingness to pay

# --- Machine's Pricing Vector 'r' ---
# This is a fallback pricing vector, when we don't feed u_hat to calculate_price
PRICING_R = np.zeros(D)

# --- 2. Define Sampling Functions ---
# def context_sampler() -> np.ndarray:
#     """Samples a customer's context vector from a uniform distribution."""
#     return np.random.uniform(low=0.0, high=1.0, size=D)

def context_sampler() -> np.ndarray:
    """Samples a customer's context vector uniformly from the unit ball."""
    return np.abs(unit_ball_rejection_sample(D))

def rental_sampler() -> float:
    """Samples a customer's desired rental duration from an exponential distribution."""
    return np.random.exponential(scale=10.0)

def interarrival_sampler() -> float:
    """Samples the time until the next customer arrives."""
    return np.random.exponential(scale=5.0)

# --- 3. Instantiate Simulation Components ---
print("Initializing simulation components...")

usage_exp_hazard_model = ExponentialHazard(lambda_val=LAMBDA_VAL)
# spontaneous_exp_hazard_model = None # ExponentialHazard(lambda_val=0.01)

customer_gen = CustomerGenerator(
    d=D,
    context_sampler=context_sampler,
    rental_sampler=rental_sampler,
    interarrival_sampler=interarrival_sampler
)

centroid_params = {
    # 'num_samples': 2000,
    # 'thin': None,
    # 'burn_in': 500 * D ** 2,
    # 'tol': 1e-4,
    # 'rho_target': 0.01
}

termination_rule = lambda diameter: diameter < 0.0005  # Sufficiently equivalent condition

projected_volume_learner = ProjectedVolumeLearner(
    T=NUM_CUSTOMERS, 
    d=D, 
    centroid_params=centroid_params,
    incentive_constant=1.1, # Anything > 1 works due to unit ball context and utility
    termination_rule=termination_rule,
)

mdp_params = {
    'replacement_cost': 1.5,   # Cost to replace the machine
    'failure_cost': 0.75,      # Additional penalty for in-service failure
    'holding_cost_rate': 0.02,   # Cost per unit of idle time
    'gamma': 0.999,             # Discount factor
    'learning_rate': 1e-3,      # Learning rate for the Adam optimizer
    'target_update_freq': 10    # How often to update the target network (in iterations)
}

training_hyperparams = {
    'num_iterations': 50, # Number of training iterations per policy update
    'dataset_size': 500000,      # Number of transitions to generate for the offline dataset
    'batch_size': 2048           # Batch size for training
}

policy_params = { # Irrelevant parameters are automatically ignored, but kept for clarity
    'type': 'softmax',
    'tau': 1.0,
    'epsilon': 0.1,
}

# Instantiate the Simulator with the new parameters
simulator = Simulator(
    d=D,
    T=NUM_CUSTOMERS,
    
    theta_true=THETA_TRUE,
    utility_true=UTILITY_TRUE,
    pricing_r=PRICING_R,
    
    usage_hazard_model=usage_exp_hazard_model,
    customer_generator=customer_gen,
    projected_volume_learner=projected_volume_learner,  # Use default ProjectedVolumeLearner
    
    mdp_params=mdp_params,
    training_hyperparams=training_hyperparams,
    policy_params=policy_params,
    policy_update_threshold=5,
    time_normalize=True,
)

# --- 4. Run the Simulation ---
# simulator.projected_volume_learner.is_terminated = True
simulation_data = simulator.run(num_customers=NUM_CUSTOMERS)
degradation_df = pd.DataFrame(simulator.degradation_history)
simulation_df = pd.DataFrame(simulator.history)

# Calculate statistics
total_rentals = (df['feedback'] != -1).sum()
total_rejections = (df['feedback'] == -1).sum()
total_breakdowns = (df['feedback'] == 1).sum()

print("\n--- Simulation Results ---")
print(f"Total customers simulated: {len(df)}")
print(f"Total rentals accepted: {total_rentals}")
print(f"Total price rejections: {total_rejections}")
print(f"Total breakdowns observed: {total_breakdowns}")

# Save the data to a file for later use
# df.to_csv("simulation_data.csv", index=False)
# print("\nSimulation data saved to simulation_data.csv")

degradation_learner = DegradationLearner(d=D, initial_theta=np.zeros(D))

degradation_learner.fit(degradation_df)
degradation_learner.get_theta()



### Training policy under perfect information

class PerfectDegradationLearner:
    def __init__(self, d, theta_true=THETA_TRUE):
        self.d = d
        self.theta_true = theta_true
        self.hazard_model = usage_exp_hazard_model  # Placeholder, not used
        
    def get_theta(self):
        return self.theta_true
    
    def cum_baseline(self, t):
        return self.hazard_model.Lambda_0(t)
    
    def inverse_cum_baseline(self, u):
        return self.hazard_model.Lambda_0_inverse(u)
    
perfect_degradation_learner = PerfectDegradationLearner(
    d=D, 
    theta_true=THETA_TRUE,
    hazard_model=usage_exp_hazard_model,
)
perfect_dpagent = DPAgent(
    d=D,
    u_hat=UTILITY_TRUE,
    time_normalize=True,
    degradation_learner=perfect_degradation_learner,
    customer_generator=customer_gen,
    params=mdp_params,
)

# perfect_dpagent.train(
#     num_iterations=50,
#     dataset_size=500000,
#     batch_size=1024
# )

# perfect_policy = perfect_dpagent.get_policy(
#     {'type': 'greedy'}
# )

perfect_dpagent.q_network.load_state_dict(
    torch.load('perfect_dpagent_q_network.pth', map_location=perfect_dpagent.device)
)
perfect_dpagent.q_network.eval()

policy = perfect_dpagent.get_policy(
    {'type': 'greedy', 'epsilon': 0.0, 'tau': 1.0}
)

num_repeat = 10
histories = []

for _ in range(num_repeat):
    history = simulator.run_full_exploit(10000, policy)
    history = pd.DataFrame(history)
    
    history['net_profit'] = history['profit'] + history['loss']
    # calculate cumulative profit and loss
    history['cumulative_net_profit'] = history['net_profit'].cumsum()
    histories.append(history)
