import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any
from raas.hazard_models import HazardModel
from raas.optimized_utility_learner import ProjectedVolumeLearner, diam
from raas.degradation_learner import DegradationLearner
from raas.policy import Policy
import logging
import pickle
from tqdm import tqdm


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

    def reset(self, last_breakdown_time: float=0.0):
        self.last_breakdown_time: float = last_breakdown_time
        self.accumulated_context = np.zeros(self.d)
        self.cumulative_active_time = 0.0

    def calculate_price(self, customer_context, u_estimate=None) -> float:
        """Calculates the price for a given customer."""
        if u_estimate is not None:
            return customer_context @ u_estimate - self.price_eps
        return customer_context @ self.pricing_r

    def record_survival(self, context, rental_duration):
        """Updates the accumulated context after a successful rental."""
        self.accumulated_context += context
        self.cumulative_active_time += rental_duration

    def get_age(self, current_time: float) -> float:
        """Returns the machine's age since the last breakdown."""
        return current_time - self.last_breakdown_time

    def get_state_summary(self):
        """Returns the core components of the machine's state."""
        return self.accumulated_context, self.cumulative_active_time


class Simulator:
    """Orchestrates the machine rental simulation."""
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

        mdp_params: Dict[str, Any] = {},
        policy_update_threshold: int = 5,
        price_eps: float = 1e-2,
    ):
        self.d = d
        self.T = T
        self.theta_true = theta_true
        self.utility_true = utility_true
        self.usage_hazard_model = usage_hazard_model
        self.customer_generator = customer_generator

        self.projected_volume_learner = projected_volume_learner
        self.mdp_params = mdp_params
        self.policy_update_threshold = policy_update_threshold

        self.machine = Machine(d, pricing_r, price_eps)
        self.calendar_time: float = 0.0

        self.history = []
        self.degradation_history = []
        self.breakdowns_since_last_update = 0
        self.seen_breakdowns = 0
        self.theta_updates = []
        self.utility_updates = []
        self.curr_customer_idx = 0
        self.degradation_learner = None

    def _update_policy(self, customer_idx):
        """Re-estimates degradation parameters and updates the policy."""
        logging.info("Re-estimating parameters and updating policy...")
        if not self.degradation_history:
            logging.warning("Cannot update policy, no degradation history yet.")
            return

        # 1. Learn degradation parameters from the history
        self.degradation_learner = DegradationLearner(d=self.d)
        df_degradation = pd.DataFrame(self.degradation_history)
        self.degradation_learner.fit(df_degradation)
        logging.info(f"Theta updated. New theta_hat: {self.degradation_learner.get_theta().round(3)}")
        self.theta_updates.append({
            "customer_idx": customer_idx,
            "theta_hat": self.degradation_learner.get_theta().copy()
        })

        # 2. Update the policy with new estimates
        u_hat = self.projected_volume_learner.get_estimate()
        self._policy.update(u_hat, self.degradation_learner)
        self.breakdowns_since_last_update = 0

        logging.info("Policy updated.")

    def run(self, num_customers: int, policy: Policy) -> List[Dict[str, Any]]:
        """Runs the simulation for a specified number of customers.

        Args:
            num_customers: Number of customer arrivals to simulate.
            policy: A Policy object that makes rental and replacement decisions.
                    For oracle runs with no re-estimation, set
                    policy_update_threshold=None on the Simulator and pre-initialize
                    the policy via policy.update(u_hat, degradation_learner).

        Returns:
            List of event dicts (the simulation history).
        """
        logging.info(f"Starting simulation for {num_customers} customers...")
        self._policy = policy
        self._first_update_done = False

        pbar = tqdm(range(num_customers))
        for i in pbar:
            # 1. Generate a new customer
            customer = self.customer_generator.generate()
            self.calendar_time += customer['interarrival_time']
            arrival_time = self.calendar_time
            self.curr_customer_idx += 1
            
            # Notify policy of every customer (for empirical stats)
            self._policy.on_customer_observed(
                customer['context'], customer['desired_duration']
            )

            self.history.append({
                "event_type": "customer_arrival",
                "customer_id": self.curr_customer_idx,
                "calendar_time": self.calendar_time,
                "profit": 0,
                "loss": -self.mdp_params['holding_cost_rate'] * customer['interarrival_time'],
            })

            # Get current machine state BEFORE interaction
            X_before, t_before = self.machine.get_state_summary()

            is_exploration_done = (
                self.projected_volume_learner.is_terminated
                and (self.seen_breakdowns > 1)
            )

            if not is_exploration_done:
                # --- Exploration phase: learn utility via ProjectedVolumeLearner ---
                u_learn_data = self.projected_volume_learner.update(
                    customer['context'], self.utility_true
                )

                if len(u_learn_data) == 0:
                    logging.warning("No data from utility learner; skipping customer.")
                    self.history.pop()
                    self.calendar_time -= customer['interarrival_time']
                    continue

                centroid = u_learn_data['centroid']
                rented = u_learn_data['rented']
                profit = u_learn_data['profit']
                event_type = ("rental_during_exploration" if rented
                              else "price_rejection_during_exploration")

                self.utility_updates.append({
                    "customer_idx": self.curr_customer_idx,
                    "u_hat": centroid,
                })
                _, diameter = self.projected_volume_learner.check_termination(
                    customer['context']
                )
                logging.info(
                    f"Customer {self.curr_customer_idx}: Diameter: {diameter:.4f}"
                )

            else:
                # --- Exploitation phase: use policy for decisions ---

                # First-time policy initialization when entering exploitation
                if not self._first_update_done:
                    if self.policy_update_threshold is not None:
                        logging.info(
                            f"Exploration phase completed at customer "
                            f"{self.curr_customer_idx}."
                        )
                        self._update_policy(self.curr_customer_idx)
                    self._first_update_done = True

                # Build state dict with raw vectors
                arrival_state = {
                    'X_accumulated': X_before.copy(),
                    'customer_context': customer['context'],
                    'customer_duration': customer['desired_duration'],
                    'cumulative_active_time': self.machine.cumulative_active_time,
                    'phase': 0,
                }

                action = self._policy(arrival_state)

                price = self.machine.calculate_price(
                    customer['context'],
                    self.projected_volume_learner.get_estimate()
                )
                if action == 1:
                    price += 100000.0  # Prohibitively high price to simulate shutdown

                rented = (np.dot(self.utility_true, customer['context']) >= price)
                profit = price if rented else 0.0

                if rented:
                    event_type = "rental_post_exploration"
                elif action == 1:
                    event_type = "shutdown"
                else:
                    event_type = "price_rejection_post_exploration"

            # --- Handle Outcome ---
            if not rented:
                self.history.append({
                    "event_type": event_type,
                    "customer_id": self.curr_customer_idx,
                    "calendar_time": arrival_time,
                    "profit": profit,
                    "loss": 0
                })
                continue

            # --- Rental proceeds: Calculate hazard and outcome ---
            X_total = X_before + customer['context']
            machine_age_at_rental = self.machine.cumulative_active_time
            rate = (self.usage_hazard_model.lambda_0(machine_age_at_rental)
                    * np.exp(np.dot(X_total, self.theta_true)))

            # Use true Cox model to simulate time to failure
            # TODO: if lambda_0 is not constant, use integration and inversion
            remaining_hazard = np.random.exponential(1.0)
            time_to_failure = remaining_hazard / rate if rate > 0 else np.inf

            if time_to_failure <= customer['desired_duration']:
                feedback, observed_duration = 1, time_to_failure
                self.breakdowns_since_last_update += 1
                self.seen_breakdowns += 1
                loss = -(self.mdp_params['failure_cost']
                         + self.mdp_params['replacement_cost'])
                event_type = "failure"
            else:
                feedback, observed_duration = 0, customer['desired_duration']
                loss = 0.0
                event_type = "survival"

            self.history.append({
                "event_type": event_type,
                "customer_id": self.curr_customer_idx,
                "calendar_time": self.calendar_time + observed_duration,
                "profit": profit,
                "loss": loss
            })

            self.calendar_time += observed_duration

            self.degradation_history.append({
                "start": t_before,
                "stop": t_before + observed_duration,
                "event": feedback,
                **{f"X{j}": v for j, v in enumerate(X_total)}
            })

            # --- Update machine state ---
            if feedback == 1:
                self.machine.reset(self.calendar_time)
            else:
                self.machine.record_survival(
                    customer['context'], customer['desired_duration']
                )

            # --- Post-Rental Policy Decision (Replacement or Not) ---
            if is_exploration_done:
                X_after, _ = self.machine.get_state_summary()
                departure_state = {
                    'X_accumulated': X_after.copy(),
                    'customer_context': np.zeros(self.d),
                    'customer_duration': 0.0,
                    'cumulative_active_time': self.machine.cumulative_active_time,
                    'phase': 1,
                }

                action = self._policy(departure_state)
                if action == 2:  # Replace Machine
                    self.history.append({
                        "event_type": "replacement",
                        "customer_id": self.curr_customer_idx,
                        "calendar_time": self.calendar_time,
                        "profit": 0,
                        "loss": -self.mdp_params['replacement_cost'],
                    })
                    self.machine.reset()
                    self.machine.last_breakdown_time = self.calendar_time

            # --- Check if policy update is needed ---
            if (is_exploration_done
                    and self.policy_update_threshold is not None
                    and self.breakdowns_since_last_update
                    >= self.policy_update_threshold):
                self._update_policy(self.curr_customer_idx)

        logging.info("Simulation finished.")
        return self.history

    def save(self, path):
        """Save simulation results and config for post-hoc analysis.

        Saves all analysis outputs (history, parameter convergence tracks)
        plus lightweight config and machine state. Skips unpicklable objects
        (customer_generator, projected_volume_learner, _policy).

        Args:
            path: File path without extension; '.pkl' is appended.
        """
        state = {
            # Analysis outputs
            'history': self.history,
            'theta_updates': self.theta_updates,
            'utility_updates': self.utility_updates,
            'degradation_history': self.degradation_history,

            # Config for reproducibility
            'd': self.d,
            'T': self.T,
            'theta_true': self.theta_true,
            'utility_true': self.utility_true,
            'mdp_params': self.mdp_params,
            'policy_update_threshold': self.policy_update_threshold,

            # Simulation state
            'calendar_time': self.calendar_time,
            'curr_customer_idx': self.curr_customer_idx,
            'seen_breakdowns': self.seen_breakdowns,
            'breakdowns_since_last_update': self.breakdowns_since_last_update,
            'machine_state': {
                'accumulated_context': self.machine.accumulated_context.copy(),
                'cumulative_active_time': self.machine.cumulative_active_time,
                'last_breakdown_time': self.machine.last_breakdown_time,
            },

            # Reconstruction hints
            'hazard_lambda_val': self.usage_hazard_model.lambda_val,
            'u_hat_final': (
                self.projected_volume_learner.get_estimate()
                if self.projected_volume_learner.is_terminated
                else None
            ),
        }
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"Simulator saved to {path}.pkl")

    @classmethod
    def load(cls, path):
        """Load a saved simulation for analysis.

        Reconstructs the Simulator with all analysis-relevant fields
        (history, theta_updates, utility_updates, degradation_history)
        and lightweight state. Unpicklable runtime objects
        (customer_generator, projected_volume_learner, _policy) are set
        to None — call run() is not supported on a loaded instance.

        Args:
            path: File path without extension; '.pkl' is appended.

        Returns:
            A Simulator instance populated with saved data.
        """
        with open(path + '.pkl', 'rb') as f:
            state = pickle.load(f)

        sim = object.__new__(cls)

        # Config
        sim.d = state['d']
        sim.T = state['T']
        sim.theta_true = state['theta_true']
        sim.utility_true = state['utility_true']
        sim.mdp_params = state['mdp_params']
        sim.policy_update_threshold = state['policy_update_threshold']

        # Analysis outputs
        sim.history = state['history']
        sim.theta_updates = state['theta_updates']
        sim.utility_updates = state['utility_updates']
        sim.degradation_history = state['degradation_history']

        # Simulation state
        sim.calendar_time = state['calendar_time']
        sim.curr_customer_idx = state['curr_customer_idx']
        sim.seen_breakdowns = state['seen_breakdowns']
        sim.breakdowns_since_last_update = state['breakdowns_since_last_update']

        # Reconstruct machine from saved scalars
        ms = state['machine_state']
        sim.machine = Machine(sim.d, np.zeros(sim.d))
        sim.machine.accumulated_context = ms['accumulated_context']
        sim.machine.cumulative_active_time = ms['cumulative_active_time']
        sim.machine.last_breakdown_time = ms['last_breakdown_time']

        # Unpicklable / runtime-only objects
        sim.usage_hazard_model = None
        sim.customer_generator = None
        sim.projected_volume_learner = None
        sim.degradation_learner = None
        sim._policy = None

        logging.info(f"Simulator loaded from {path}.pkl")
        return sim
