import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    """
    Abstract base class for all policies used by the Simulator.

    The call signature: action = policy(state) -> int

    State is a dict with keys:
        'X_accumulated':         np.ndarray (d,) — cumulative context since last replacement
        'customer_context':      np.ndarray (d,) — current customer's context (zeros for departure)
        'customer_duration':     float            — rental duration (0 for departure)
        'cumulative_active_time': float           — machine active time since last replacement
        'phase':                 int              — 0 (arrival) or 1 (departure)

    Actions:
        0 = accept / offer price   (arrival only)
        1 = reject / shutdown      (arrival only)
        2 = replace machine        (departure only)
        3 = do not replace         (departure only)
    """

    @abstractmethod
    def __call__(self, state: dict) -> int:
        """Returns an action given the current state."""
        ...

    def update(self, u_hat: np.ndarray, degradation_learner) -> None:
        """Called when the simulator re-estimates parameters.

        Args:
            u_hat: Updated utility estimate vector (d,).
            degradation_learner: Fitted DegradationLearner with get_theta(),
                                 cum_baseline(), and predict_failure_prob().
        """
        pass

    def on_customer_observed(self, context: np.ndarray, duration: float) -> None:
        """Called after every customer is generated (before accept/reject decision).

        Args:
            context: Customer context vector x_k (d,).
            duration: Customer desired rental duration T_k.
        """
        pass


class MyopicBasePolicy(Policy):
    """
    Shared logic for both myopic baselines.

    Implements:
      - Myopic rental accept/reject: accept if pi_myopic > 0
      - Epsilon-greedy exploration with decay
      - Storage of u_hat, degradation_learner references

    Subclasses override the departure decision only.
    """

    def __init__(self, mdp_params: dict,
                 initial_epsilon: float = 0.10,
                 decay_rate: float = 0.95,
                 min_epsilon: float = 0.001):
        self.failure_cost = mdp_params['failure_cost']
        self.replacement_cost = mdp_params['replacement_cost']

        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self._policy_step = 0

        self.u_hat = None
        self.degradation_learner = None

    @property
    def current_epsilon(self) -> float:
        return max(self.min_epsilon,
                   self.initial_epsilon * (self.decay_rate ** self._policy_step))

    def update(self, u_hat, degradation_learner):
        self.u_hat = u_hat
        self.degradation_learner = degradation_learner
        self._policy_step += 1

    def _myopic_arrival_action(self, state: dict) -> int:
        """Accept if expected single-rental profit > 0.

        pi_myopic = u_hat @ x - p_fail * (F + R)
        """
        X = state['X_accumulated']
        x = state['customer_context']
        T = state['customer_duration']
        t = state['cumulative_active_time']

        revenue = np.dot(self.u_hat, x)
        p_fail = self.degradation_learner.predict_failure_prob(X, x, T, t_age=t)
        pi_myopic = revenue - p_fail * (self.failure_cost + self.replacement_cost)
        return 0 if pi_myopic > 0 else 1

    def __call__(self, state: dict) -> int:
        phase = state['phase']

        if phase == 0:  # Arrival
            if np.random.rand() < self.current_epsilon:
                return np.random.choice([0, 1])
            return self._myopic_arrival_action(state)
        else:  # Departure
            if np.random.rand() < self.current_epsilon:
                return np.random.choice([2, 3])
            return self._departure_decision(state)

    @abstractmethod
    def _departure_decision(self, state: dict) -> int:
        """Subclass-specific departure logic. Returns 2 (replace) or 3 (keep)."""
        ...
