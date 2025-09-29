import numpy as np
import pandas as pd

from raas.neural_policy import DPAgent
# from discrete_policy import DiscretizedDPAgent
from raas.simulation import Simulator, CustomerGenerator
from raas.hazard_models import ExponentialHazard
from raas.utility_learner import ProjectedVolumeLearner
from raas.degradation_learner import DegradationLearner
from raas.discrete_policy import DiscretizedDPAgent
from datetime import datetime
from pytz import timezone

import logging
logging.basicConfig(level=logging.INFO)

from raas.config import (
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
import argparse

if __name__ == "__main__":
    # use argparse to get `skip_training` from command line arguments. It can be either True or False.
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_training', type=bool, default=False, help='Whether to skip training the policy.')
    args = parser.parse_args()
    skip_training = args.skip_training
    
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
    
    if skip_training:
        logging.info("Skipping policy training as per user request.")
        simulator.projected_volume_learner.centroids.append(UTILITY_TRUE)
        simulator.projected_volume_learner.is_terminated = True
        simulator.seen_breakdowns = 2

        degradation_learner = DegradationLearner(d=simulator.d)
        degradation_learner.theta = np.ones(D) * 0.1
        degradation_learner.cum_baseline = lambda x: LAMBDA_VAL * x
        degradation_learner.inverse_cum_baseline = lambda y: y / LAMBDA_VAL
        simulator.degradation_learner = degradation_learner
        
        dp_agent = DiscretizedDPAgent(
            N=training_hyperparams['N'], # grid sizes [cum_context, context, duration, active_time]
            max_cumulative_context=training_hyperparams['max_cumulative_context'],
            # max_active_time=training_hyperparams['max_active_time'],
            u_hat=UTILITY_TRUE,
            degradation_learner=degradation_learner,
            customer_generator=customer_gen,
            params=mdp_params,
        )
        
        # dp_agent._precompute_dynamics(num_samples=50000)
        dp_agent.run_value_iteration(100)

        simulator.dp_agent = dp_agent
        simulator.optimal_policy = dp_agent.get_policy(simulator.policy_type)
        simulator.breakdowns_since_last_update = 0 # Reset the counter
    
    pacific_tz = timezone('America/Los_Angeles')
    current_time = datetime.now(pacific_tz).strftime("%Y%m%d_%H%M%S")

    # simulator.projected_volume_learner.is_terminated = True
    simulation_data = simulator.run(num_customers=NUM_CUSTOMERS)
    degradation_df = pd.DataFrame(simulator.degradation_history)
    simulation_df = pd.DataFrame(simulator.history)

    degradation_df.to_csv(f'data/degradation_data_{current_time}.csv', index=False)
    simulation_df.to_csv(f'data/simulation_data_{current_time}.csv', index=False)
    simulator.save(f'models/simulator_{current_time}')