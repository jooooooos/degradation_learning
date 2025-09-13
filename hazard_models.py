from abc import ABC, abstractmethod

class HazardModel(ABC):
    """
    Abstract Base Class for baseline hazard functions.
    
    This class defines the interface that all hazard models must implement,
    ensuring they can be used interchangeably by the simulator.
    """
    
    @abstractmethod
    def lambda_0(self, t: float) -> float:
        """Calculates the baseline hazard rate lambda_0(t) at machine age t."""
        pass

    @abstractmethod
    def Lambda_0(self, t: float) -> float:
        """Calculates the cumulative baseline hazard Lambda_0(t) = integral_0^t lambda_0(s) ds."""
        pass

    @abstractmethod
    def Lambda_0_inverse(self, y: float) -> float:
        """Calculates the inverse of the cumulative baseline hazard function."""
        pass


class ExponentialHazard(HazardModel):
    """
    A concrete implementation of a HazardModel for an exponential baseline hazard.
    
    The hazard rate is constant: lambda_0(t) = lambda_val.
    
    Args:
        lambda_val (float): The constant hazard rate.
    """
    def __init__(self, lambda_val: float):
        if lambda_val <= 0:
            raise ValueError("Lambda value must be positive.")
        self.lambda_val = lambda_val

    def lambda_0(self) -> float:
        """Returns the constant baseline hazard rate."""
        return self.lambda_val

    def Lambda_0(self, t: float) -> float:
        """Calculates the cumulative baseline hazard: lambda * t."""
        return self.lambda_val * t

    def Lambda_0_inverse(self, y: float) -> float:
        """Calculates the inverse of the cumulative baseline hazard: y / lambda."""
        if y < 0:
            raise ValueError("Input to inverse cumulative hazard must be non-negative.")
        return y / self.lambda_val
