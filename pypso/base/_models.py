from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseOptimizer(ABC):
    """Base class for particle swarm optimizers.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    objective : object
        A function handle to optimize.

    n_particles : int
        The number of particles in the swarm.

    x_bounds : iterable
        Iterable of iterables, where the ith element contains the lower and 
        upper bounds (inclusive) for variable i. For example, a list of tuples 
        such as [(1, 1), (1, 10)] would represent x1 with bounds [1, 1] and 
        x2 with bounds [1, 10].

    w_bounds : iterable
        Two-item iterable containing the lower and upper bounds (inclusive) for 


    Attributes
    ----------
    ADD HERE
    """
    @abstractmethod
    def __init__(self, 
                 objective: Any, 
                 n_particles: int, 
                 x_bounds: Iterable,
                 w_bounds: Iterable,
                 v_bounds: Iterable, 
                 max_iter: int = 100, 
                 phi_p: float = 1,
                 phi_g: float = 1,
                 w_decay: bool = True) -> None:
        pass


    @abstractmethod
    def __str__(self) -> None:
        """ADD.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass