import numpy as np
from raas.policy import MyopicBasePolicy
from raas.config import LAMBDA_VAL

class MyopicHeuristicPolicy(MyopicBasePolicy):
    """
    Baseline B: Myopic + Heuristic Replacement.

    - Arrival: same myopic accept/reject as Baseline A.
    - Departure: replace when the average customer is no longer profitable
      at the current degradation level.

    Maintains running empirical means of:
      - x_bar_emp: mean customer context vector (d,)
      - T_bar_emp: mean customer rental duration (scalar)

    These are computed over ALL observed customers (including rejected ones),
    updated via on_customer_observed() hook.
    """

    def __init__(self, mdp_params: dict, d: int,
                 initial_epsilon: float = 0.10,
                 decay_rate: float = 0.95,
                 min_epsilon: float = 0.001):
        super().__init__(mdp_params, initial_epsilon, decay_rate, min_epsilon)
        self.d = d
        self._customer_count = 0
        self._sum_context = np.zeros(d)
        self._sum_duration = 0.0

    def on_customer_observed(self, context: np.ndarray, duration: float):
        self._customer_count += 1
        self._sum_context += context
        self._sum_duration += duration

    @property
    def x_bar_emp(self) -> np.ndarray:
        if self._customer_count == 0:
            return np.zeros(self.d)
        return self._sum_context / self._customer_count

    @property
    def T_bar_emp(self) -> float:
        if self._customer_count == 0:
            return 1.0
        return self._sum_duration / self._customer_count

    @property
    def replacement_threshold(self) -> float:
        """
        For external evaluadion.
        Effective replacement threshold satisfies,
        
        u_hat @ x_bar_emp - p_fail(threshold) * (F + R) = 0
        p_fail(threshold) = 1 - exp(-lambda_hat * exp(threshold + theta_hat @ x_bar_emp) * T_bar_emp)
        """
        result = self.u_hat @ self.x_bar_emp / (self.failure_cost + self.replacement_cost)
        result = np.log(1 - result) # == -lambda_hat * exp(threshold + theta_hat @ x_bar_emp) * T_bar_emp
        result = result / (-LAMBDA_VAL * self.T_bar_emp) # use true lambda for interpretability
        result = np.log(result) - (self.degradation_learner.get_theta() @ self.x_bar_emp)
        return result

    def _departure_decision(self, state: dict) -> int:
        """Replace if the average customer would be unprofitable at current degradation.

        Computes:
            p_fail_avg using current X_accumulated and empirical average customer
            pi_avg = u_hat @ x_bar_emp - p_fail_avg * (F + R)
            Replace if pi_avg <= 0.
        """
        if self.degradation_learner is None or self._customer_count == 0:
            return 3  # Cannot evaluate yet, keep machine

        X = state['X_accumulated']
        t = state['cumulative_active_time']

        x_bar = self.x_bar_emp
        T_bar = self.T_bar_emp

        cu_avg = np.dot(self.u_hat, x_bar)
        p_fail_avg = self.degradation_learner.predict_failure_prob(
            X, x_bar, T_bar, t_age=t
        )
        pi_avg = cu_avg - p_fail_avg * (self.failure_cost + self.replacement_cost)

        return 2 if pi_avg <= 0 else 3
