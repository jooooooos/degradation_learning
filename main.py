import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from simulation import Simulator, CustomerGenerator
from hazard_models import ExponentialHazard
from utility_learner import ProjectedVolumeLearner, diam
from degradation_learner import DegradationLearner

from utils import unit_ball_rejection_sample, correct_signs
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import logging
logging.basicConfig(level=logging.INFO)

# --- 1. Simulation Configuration ---
D = 5                                  # Dimension of context vectors
LAMBDA_VAL = 0.001                     # Baseline hazard constant
NUM_CUSTOMERS = 100                   # Total number of customers to simulate

# Set a random seed for reproducibility
np.random.seed(42)

# Ground truth vectors
THETA_TRUE = np.array([0.5, 0.2, 0.1, 0.3, 0.4])    # For degradation
UTILITY_TRUE = context_sampler()  # For customer's willingness to pay

# --- Machine's Pricing Vector 'r' ---
# You can change this to test different pricing strategies.
# Case 1: A non-zero pricing strategy
# PRICING_R = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
# Case 2: Zero price (free rentals), guaranteeing 100% acceptance
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
    return np.random.exponential(scale=20.0)

def interarrival_sampler() -> float:
    """Samples the time until the next customer arrives."""
    return np.random.exponential(scale=5.0)

# --- 3. Instantiate Simulation Components ---
print("Initializing simulation components...")

usage_exp_hazard_model = ExponentialHazard(lambda_val=LAMBDA_VAL)
spontaneous_exp_hazard_model = None

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

termination_rule = lambda diameter: diameter < 0.01  # Example custom termination rule
projected_volume_learner = ProjectedVolumeLearner(
    T=NUM_CUSTOMERS, 
    d=D, 
    centroid_params=centroid_params,
    termination_rule=termination_rule,
)

# Instantiate the Simulator with the new parameters
simulator = Simulator(
    d=D,
    T=NUM_CUSTOMERS,
    theta=THETA_TRUE,
    utility_true=UTILITY_TRUE,
    pricing_r=PRICING_R,
    usage_hazard_model=usage_exp_hazard_model,
    spontaneous_hazard_model=spontaneous_exp_hazard_model,
    customer_generator=customer_gen,
    projected_volume_learner=projected_volume_learner,  # Use default ProjectedVolumeLearner
)

# --- 4. Run the Simulation ---
simulation_data = simulator.run(num_customers=NUM_CUSTOMERS)
df = pd.DataFrame(simulation_data)

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

degradation_learner.fit(df)
degradation_learner.get_theta()