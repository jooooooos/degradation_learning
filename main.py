import numpy as np
import pandas as pd

from policy import DPAgent
# from discrete_policy import DiscretizedDPAgent
from simulation import Simulator, CustomerGenerator
from hazard_models import ExponentialHazard
from utility_learner import ProjectedVolumeLearner
# from degradation_learner import DegradationLearner
from datetime import datetime
from pytz import timezone

import logging
logging.basicConfig(level=logging.INFO)

from config import (
    context_sampler,
    rental_sampler,
    interarrival_sampler,
    
    D,
    LAMBDA_VAL,
    NUM_CUSTOMERS,
    THETA_TRUE,
    UTILITY_TRUE,
    PRICING_R,
    
    centroid_params,
    termination_rule,
    
    mdp_params,
    training_hyperparams,
    incentive_constant,
    
    policy_type,
    policy_kwargs,
)

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # ---                   Initialize Model Instances                        --- #
    # --------------------------------------------------------------------------- #
    usage_exp_hazard_model = ExponentialHazard(lambda_val=LAMBDA_VAL)

    customer_gen = CustomerGenerator(
        d=D,
        context_sampler=context_sampler,
        rental_sampler=rental_sampler,
        interarrival_sampler=interarrival_sampler
    )

    projected_volume_learner = ProjectedVolumeLearner(
        T=NUM_CUSTOMERS, 
        d=D, 
        centroid_params=centroid_params,
        incentive_constant=incentive_constant,
        termination_rule=termination_rule,
    )

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
        discrete_dp=True,
        policy_type=policy_type,
        training_hyperparams=training_hyperparams,
        policy_kwargs=policy_kwargs,
        policy_update_threshold=100,
        time_normalize=True,
    )
    
    pacific_tz = timezone('America/Los_Angeles')
    current_time = datetime.now(pacific_tz).strftime("%Y%m%d_%H%M%S")

    # simulator.projected_volume_learner.is_terminated = True
    simulation_data = simulator.run(num_customers=NUM_CUSTOMERS)
    degradation_df = pd.DataFrame(simulator.degradation_history)
    simulation_df = pd.DataFrame(simulator.history)

    degradation_df.to_csv(f'data/degradation_data_{current_time}.csv', index=False)
    simulation_df.to_csv(f'data/simulation_data_{current_time}.csv', index=False)
    simulator.save(f'models/simulator_{current_time}')