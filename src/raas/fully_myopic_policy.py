from raas.policy import MyopicBasePolicy


class FullyMyopicPolicy(MyopicBasePolicy):
    """
    Baseline A: Fully Myopic.

    - Arrival: Accept if single-rental expected profit > 0 (myopic).
    - Departure: NEVER voluntarily replace (always action 3).

    Expected pathology: accumulates degradation indefinitely, eventually
    rejects all customers because p_fail becomes too high. Only replaced
    when epsilon-greedy exploration triggers an accept that causes failure.
    """

    def _departure_decision(self, state: dict) -> int:
        return 3  # Never voluntarily replace
