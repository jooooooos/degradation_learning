import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any
from hazard_models import HazardModel
from utility_learner import ProjectedVolumeLearner, diam
from degradation_learner import DegradationLearner
from policy import DPAgent
import logging
from tqdm import tqdm, trange

class CustomerGenerator:
    """Generates new customers with their specific attributes."""
    def __init__(
        self,
        d: int,
        context_sampler: Callable[[], np.ndarray],
        rental_sampler: Callable[[], float],
        interarrival_sampler: Callable[[], float]
    ):
        self.d = d
        self.context_sampler = context_sampler
        self.rental_sampler = rental_sampler
        self.interarrival_sampler = interarrival_sampler

    def generate(self) -> Dict[str, Any]:
        """Creates a new customer with context, desired duration, and interarrival time."""
        return {
            "context": self.context_sampler(),
            "desired_duration": self.rental_sampler(),
            "interarrival_time": self.interarrival_sampler()
        }

class Machine:
    """Manages the state of a single machine instance and its pricing."""
    def __init__(self, d: int, pricing_r: np.ndarray, price_eps: float=1e-2):
        self.d = d
        self.pricing_r = pricing_r
        self.price_eps = price_eps
        self.reset()
        
    def reset(self):
        self.last_breakdown_time: float = 0.0
        self.accumulated_context = np.zeros(self.d)
        self.cumulative_idle_time = 0.0
        self.cumulative_hazard: float = 0.0
        self.E: float = np.random.exponential(1.0)
                
    def calculate_price(self, customer_context: np.ndarray, u_estimate: float) -> float:
        """Calculates the price for a given customer."""
        if u_estimate is not None:
            return customer_context @ u_estimate - self.price_eps
        return customer_context @ self.r
    
    def record_survival(self, context, idle_time):
        """Updates the accumulated context after a successful rental."""
        self.accumulated_context += context
        # self.machine_age += duration + idle_time
        self.cumulative_idle_time += idle_time

    def get_age(self, current_time: float) -> float:
        """Returns the machine's age since the last breakdown."""
        return current_time - self.last_breakdown_time

    def get_state_summary(self):
        """Returns the core components of the machine's state."""
        return self.accumulated_context, self.cumulative_idle_time

class Simulator:
    """Orchestrates the machine rental simulation."""
    # def __init__(
    #     self,
    #     d: int,
    #     T,
    #     theta: np.ndarray,
    #     utility_true: np.ndarray,
    #     pricing_r: np.ndarray,
    #     hazard_model: HazardModel,
    #     customer_generator: CustomerGenerator,
    #     projected_volume_learner: ProjectedVolumeLearner=None,
    # ):
    def __init__(
        self,
        T: int,
        d: int,
        
        theta_true: np.ndarray,
        utility_true: np.ndarray,
        pricing_r: np.ndarray,

        usage_hazard_model: HazardModel,
        customer_generator: CustomerGenerator,
        projected_volume_learner: ProjectedVolumeLearner,
        
        spontaneous_hazard_model: HazardModel=None,
        mdp_params: Dict[str, Any]=None,
        training_hyperparams: Dict[str, Any]=None,
        policy_update_threshold: int=5,
    ):
        self.d = d
        self.T = T
        self.theta_true = theta_true
        self.utility_true = utility_true        
        self.usage_hazard_model = usage_hazard_model
        self.customer_generator = customer_generator
        
        self.projected_volume_learner = projected_volume_learner
        self.mdp_params = mdp_params
        self.training_hyperparams = training_hyperparams

        self.machine = Machine(d, pricing_r)
        self.calendar_time: float = 0.0
        self.optimal_policy = None
            
        # Pass the pricing vector 'r' to the machine
        self.history = []
        self.degradation_history = []
        self.policy_update_threshold = policy_update_threshold
        self.breakdowns_since_last_update = 0
        
    def _update_policy(self):
        """Trains (or re-trains) the DP Agent and updates the policy."""
        logging.info("Updating optimal policy...")
        if not self.degradation_history:
            logging.warning("Cannot update policy, no degradation history yet.")
            return

        # 1. Learn degradation parameters from the history
        self.degradation_learner = DegradationLearner(d=self.d)
        df_degradation = pd.DataFrame(self.degradation_history)
        self.degradation_learner.fit(df_degradation)
        
        # 2. Instantiate and train the DP agent
        u_hat = self.projected_volume_learner.get_estimate()
        dp_agent = DPAgent(
            d=self.d,
            u_hat=u_hat,
            degradation_learner=self.degradation_learner,
            customer_generator=self.customer_generator,
            params=self.mdp_params
        )
        dp_agent.train(**self.training_hyperparams)
        
        self.optimal_policy = dp_agent.get_policy()
        self.breakdowns_since_last_update = 0 # Reset the counter
        logging.info(f"Policy updated. New theta_hat: {self.degradation_learner.get_theta().round(3)}")
        
    def run(self, num_customers: int) -> List[Dict[str, Any]]:
        """Runs the simulation for a specified number of customers."""
        logging.info(f"Starting simulation for {num_customers} customers...")
        
        pbar = tqdm(range(num_customers))
        for i in pbar:
            # 1. Generate a new customer
            customer = self.customer_generator.generate()
            self.calendar_time += customer['interarrival_time']
            arrival_time = self.calendar_time

            # Get current machine state BEFORE interaction
            X_before, I_before = self.machine.get_state_summary()

            is_exploration_done = self.projected_volume_learner.is_terminated
            
            if not is_exploration_done:
                # 2. If not done exploring, offer price and see if customer rents
                u_learn_data = self.projected_volume_learner.update(customer['context'], self.utility_true)
                rented = u_learn_data['rented']
                profit = u_learn_data['profit']
                
                done, diameter = self.projected_volume_learner.check_termination(
                    customer['context']
                )
                logging.info(f"Customer {i+1}: Diameter: {diameter:.4f}")
                if done:
                    logging.info(f"Exploration phase completed at customer {i+1}.")
                
            else:
                # Exploration is over, use the optimal policy
                if self.optimal_policy is None:
                    self._update_policy() # First time policy setup

                arrival_state = np.concatenate([
                    X_before, 
                    customer['context'], 
                    [customer['desired_duration'], I_before, 0.0]
                ])
                action = self.optimal_policy(arrival_state)

                price = self.machine.calculate_price(
                    customer['context'], 
                    self.projected_volume_learner.get_estimate()
                )
                if action == 1:
                    price += 100000.0 # Prohibitively high price to simulate shutdown
                
                rented = (np.dot(self.utility_true, customer['context']) >= price)
                profit = price if rented else 0.0
            
            # --- Handle Outcome ---
            if not rented:
                self.history.append({
                    "event_type": "price_rejection",
                    "customer_id": i + 1,
                    "calendar_time": arrival_time,
                    "profit": profit,
                })
                continue
            
            # --- Rental proceeds: Calculate hazard and outcome ---
            X_total = X_before + customer['context']
            machine_age_at_rental = self.machine.get_age(arrival_time)
            rate = self.usage_hazard_model.lambda_0() * np.exp(np.dot(X_total, self.theta_true))

            remaining_hazard = self.machine.E - self.machine.cumulative_hazard
            time_to_failure = remaining_hazard / rate if rate > 0 else np.inf
            
            if time_to_failure <= customer['desired_duration']:
                feedback, observed_duration = 1, time_to_failure
                self.machine.cumulative_hazard += rate * observed_duration
                self.breakdowns_since_last_update += 1
            else:
                feedback, observed_duration = 0, customer['desired_duration']
                self.machine.cumulative_hazard += rate * observed_duration
            
            self.calendar_time += observed_duration
            self.history.append({
                "event_type": "rental", "customer_id": i + 1,
                "calendar_time": arrival_time, "observed_duration": observed_duration,
                "feedback": feedback, "profit": profit,
            })

            self.degradation_history.append({
                # adjustment by idle time is necessary to get correct degradation under usage only
                "start": machine_age_at_rental - I_before,
                "stop": machine_age_at_rental - I_before + observed_duration,
                "event": feedback,
                **{f"X{j}": v for j, v in enumerate(X_total)}
            })
            
            # --- Update machine state ---
            if feedback == 1:
                self.machine.reset()
                self.machine.last_breakdown_time = self.calendar_time
            else:
                self.machine.record_survival(customer['context'], customer['interarrival_time'])
                
            # --- Post-Rental Policy Decision (Replacement or Not) ---
            if is_exploration_done:
                X_after, I_after = self.machine.get_state_summary()
                departure_state = np.concatenate([
                    X_after, np.zeros(self.d), 
                    [0.0, I_after, 1.0]
                ])
                action = self.optimal_policy(departure_state)
                if action == 2: # Replace Machine
                    self.history.append({
                        "event_type": "replacement",
                        "customer_id": i + 1,
                        "calendar_time": self.calendar_time,
                        "profit": -self.mdp_params['replacement_cost'],
                    })
                    self.machine.reset()
                    self.machine.last_breakdown_time = self.calendar_time
            
            # --- Check if policy update is needed ---
            if is_exploration_done and self.breakdowns_since_last_update >= self.policy_update_threshold:
                self._update_policy()
            
        logging.info("Simulation finished.")
        return self.history

