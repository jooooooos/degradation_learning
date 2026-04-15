import numpy as np
import logging
from raas.policy import Policy
from raas.optimized_discrete_policy import DiscretizedDPAgent


class MDPPolicy(Policy):
    """
    Wraps the DiscretizedDPAgent in the Policy interface.

    On update(): creates a new DiscretizedDPAgent, runs value iteration,
    and extracts the policy function.

    On __call__(): projects raw state vectors to the 6-tuple
    (cc, cx, cu, T, t, phase) and delegates to the DP agent's policy.
    """

    def __init__(self, training_hyperparams: dict,
                 customer_generator,
                 mdp_params: dict,
                 policy_type: str = 'decaying_epsilon_greedy',
                 initial_epsilon: float = 0.10,
                 decay_rate: float = 0.95,
                 min_epsilon: float = 0.001):
        self.training_hyperparams = training_hyperparams
        self.customer_generator = customer_generator
        self.mdp_params = mdp_params
        self.policy_type = policy_type

        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self._policy_step = 0
        self._policy_kwargs = {
            'current_epsilon': initial_epsilon,
            'decay_rate': decay_rate,
            'step': 0,
        }

        self.dp_agent = None
        self._policy_fn = None
        self.u_hat = None
        self.theta = None

    @property
    def current_epsilon(self) -> float:
        return max(self.min_epsilon,
                   self.initial_epsilon * (self.decay_rate ** self._policy_step))

    def update(self, u_hat, degradation_learner, 
               num_precompute_samples=None, 
               num_value_iterations=None,
        ):
        self.u_hat = u_hat
        self.theta = degradation_learner.get_theta()

        dp_agent = DiscretizedDPAgent(
            N=self.training_hyperparams['N'],
            max_cumulative_context=self.training_hyperparams['max_cumulative_context'],
            u_hat=u_hat,
            degradation_learner=degradation_learner,
            customer_generator=self.customer_generator,
            params=self.mdp_params,
            num_samples_precompute=self.training_hyperparams['num_precompute_samples'] if num_precompute_samples is None else num_precompute_samples,
        )

        dp_agent.run_value_iteration(num_iterations=self.training_hyperparams['num_value_iterations'] if num_value_iterations is not None else self.training_hyperparams['num_value_iterations'])

        self.dp_agent = dp_agent
        self._policy_fn = dp_agent.get_policy(self.policy_type)

        self._policy_step += 1
        self._policy_kwargs['step'] = self._policy_step

        logging.info("MDPPolicy updated via value iteration.")

    def __call__(self, state: dict) -> int:
        if self._policy_fn is None:
            raise RuntimeError(
                "MDPPolicy has not been initialized. "
                "Call update(u_hat, degradation_learner) before using, "
                "or set policy_update_threshold to a finite value so the "
                "simulator calls update() automatically."
            )

        X = state['X_accumulated']
        x = state['customer_context']
        T = state['customer_duration']
        t = state['cumulative_active_time']
        phase = state['phase']

        cc = np.dot(X, self.theta)
        cx = np.dot(x, self.theta)
        cu = np.dot(x, self.u_hat)

        projected_state = [cc, cx, cu, T, t, phase]
        return self._policy_fn(projected_state, self._policy_kwargs)
