from raas.simulation import Simulator, CustomerGenerator
import raas.config as config
from raas.degradation_learner import DegradationLearner
from raas.optimized_discrete_policy import DiscretizedDPAgent
from raas.mdp_policy import MDPPolicy
from raas.optimized_utility_learner import ProjectedVolumeLearner
from raas.fully_myopic_policy import FullyMyopicPolicy
from raas.myopic_heuristic_policy import MyopicHeuristicPolicy
import numpy as np

def pretrain_oracle(simulator, customer_gen):
    # # Lets you skip utility exploration with perfect u starting point
    simulator.policy_update_threshold = None

    simulator.projected_volume_learner.centroids.append(config.UTILITY_TRUE)
    simulator.projected_volume_learner.is_terminated = True
    simulator.seen_breakdowns = 2

    degradation_learner = DegradationLearner(d=simulator.d)
    degradation_learner.theta = config.THETA_TRUE
    degradation_learner.cum_baseline = lambda x: config.LAMBDA_VAL * x
    degradation_learner.inverse_cum_baseline = lambda y: y / config.LAMBDA_VAL
    simulator.degradation_learner = degradation_learner

    mdp_policy = MDPPolicy(
        training_hyperparams=config.training_hyperparams,
        customer_generator=customer_gen,
        mdp_params=config.mdp_params,
        policy_type='greedy',
    )

    mdp_policy.update(
        config.UTILITY_TRUE,
        degradation_learner,
        num_precompute_samples=200000,
        num_value_iterations=200,
    )
    
    return mdp_policy

def get_fresh_objects(usage_exp_hazard_model, customer_gen):
    projected_volume_learner = ProjectedVolumeLearner(
        T=config.NUM_CUSTOMERS, 
        d=config.D, 
        centroid_params=config.centroid_params,
        incentive_constant=config.incentive_constant,
        termination_rule=config.termination_rule,
    )

    # Instantiate the Simulator with the new parameters
    simulator = Simulator(
        d=config.D,
        T=config.NUM_CUSTOMERS,
        
        theta_true=config.THETA_TRUE,
        utility_true=config.UTILITY_TRUE,
        pricing_r=config.PRICING_R,
        
        usage_hazard_model=usage_exp_hazard_model,
        customer_generator=customer_gen,
        projected_volume_learner=projected_volume_learner,  # Use default ProjectedVolumeLearner
        
        mdp_params=config.mdp_params,
        policy_update_threshold=config.POLICY_UPDATE_FREQUENCY,
    )
    
    return simulator, projected_volume_learner

def run_oracle_policy(usage_exp_hazard_model, customer_gen):
    simulator, projected_volume_learner = get_fresh_objects(usage_exp_hazard_model, customer_gen)
    mdp_policy = pretrain_oracle(simulator, customer_gen)
    simulation_data = simulator.run(
        config.NUM_CUSTOMERS,
        policy=mdp_policy
    )
    return simulator, simulation_data

def run_online_policy(usage_exp_hazard_model, customer_gen):
    simulator, projected_volume_learner = get_fresh_objects(usage_exp_hazard_model, customer_gen)
    mdp_policy = MDPPolicy(
        training_hyperparams=config.training_hyperparams,
        customer_generator=customer_gen,
        mdp_params=config.mdp_params,
        policy_type='decaying_epsilon_greedy',
    )
    simulation_data = simulator.run(
        config.NUM_CUSTOMERS,
        policy=mdp_policy
    )
    return simulator, simulation_data

def run_fully_myopic_policy(usage_exp_hazard_model, customer_gen):
    simulator, projected_volume_learner = get_fresh_objects(usage_exp_hazard_model, customer_gen)
    policy = FullyMyopicPolicy(
        mdp_params=config.mdp_params,
        initial_epsilon=0.10,
        decay_rate=0.95,
        min_epsilon=0.001
    )
    
    simulation_data = simulator.run(
        config.NUM_CUSTOMERS,
        policy=policy
    )
    return simulator, simulation_data

def run_myopic_heuristic_policy(usage_exp_hazard_model, customer_gen):
    simulator, projected_volume_learner = get_fresh_objects(usage_exp_hazard_model, customer_gen)
    policy = MyopicHeuristicPolicy(
        mdp_params=config.mdp_params,
        d=config.D,
        initial_epsilon=0.10,
        decay_rate=0.95,
        min_epsilon=0.001
    )
    
    simulation_data = simulator.run(
        config.NUM_CUSTOMERS,
        policy=policy
    )
    return simulator, simulation_data

def extract_thresholds_by_t(dpagent):
    """Extract c* threshold for each active-time grid index."""
    n_t = len(dpagent.grids[4])
    thresholds = np.full(n_t, np.nan)

    for t_idx in range(n_t):
        policy_slice = dpagent.policy_departure[:, t_idx]
        replace_mask = (policy_slice == 2)
        if np.any(replace_mask):
            thresholds[t_idx] = float(dpagent.grids[0][np.argmax(replace_mask)])
        else:
            thresholds[t_idx] = float(dpagent.grids[0][-1])  # never replace within grid

    return thresholds


def get_scalar_threshold(dpagent):
    """Get single c* value (median across t indices)."""
    thresholds = extract_thresholds_by_t(dpagent)
    return float(np.median(thresholds))