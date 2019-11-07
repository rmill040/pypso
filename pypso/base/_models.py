from abc import ABC, abstractmethod
import logging
from multiprocessing import cpu_count, Pool
import numpy as np
from numpy.random import Generator, PCG64
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

__all__ = ['BasePSO']
_LOGGER = logging.getLogger(__name__)


class BasePSO(ABC):
    """Base class for particle swarm optimizers.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    n_particles : int
        The number of particles in the swarm.

    n_dimensions : int
        Number of dimensions in space.

    verbose : bool
        Controls verbosity of analysis.

    n_jobs : int
        The number of processes to use to evaluate objective function and 
        constraints.

    random_state : int
        Random seed, set to reproduce results
    """
    @abstractmethod
    def __init__(self,
                 n_particles: int,
                 n_dimensions: int,
                 verbose: bool = False,
                 n_jobs: int = 1,
                 random_state: Optional[int] = None) -> None:
        # Define attributes
        self.n_particles: int  = n_particles
        self.n_dimensions: int = n_dimensions
        self.verbose: bool     = verbose
        self.rg                = Generator(PCG64(seed=random_state))
        
        # Calculate number of jobs for parallel processing
        max_cpus: int = cpu_count()
        if n_jobs == 0:
            n_jobs = 1
        elif abs(n_jobs) > max_cpus:
            n_jobs = max_cpus
        else:
            if n_jobs < 0: n_jobs = list(range(1, cpu_count()+1))[n_jobs]
        self.n_jobs: int = n_jobs

        # Define function mapper
        map_func     = Pool(self.n_jobs).map if self.n_jobs > 1 else \
                            lambda func, x: list(map(func, x))
        self._mapper = lambda f, x: np.array(map_func(f, x))

        # Hold history of swarm results
        self.history = []

    @abstractmethod
    def __str__(self) -> str:
        """Returns name of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Name of class.
        """
        pass

    def _initialize_swarm(self, 
                          fobj: Callable[..., float],
                          lb: Union[np.ndarray, Iterable[float]],
                          ub: Union[np.ndarray, Iterable[float]],
                          fcons: Optional[Callable[..., Any]] = None, 
                          kwargs: Dict[Any, Any] = {}) -> Dict[str, Any]:
        """Initialize the swarm.
    
        Parameters
        ----------
        fobj : callable
            Function to be minimized.

        lb : iterable
            The lower bounds of the solution.

        ub : iterable
            The upper bounds of the solution.

        fcons : callable
            Function for constraints that evaluates to >= in a successfully 
            optimized problem.

        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions.

        Returns
        -------
        dict
            Key/value pairs with initialized parameters for optimization
        """
        pass

    @abstractmethod
    def optimize(self, 
                 fobj: Callable[..., float],
                 lb: Union[np.ndarray, Iterable[float]],
                 ub: Union[np.ndarray, Iterable[float]],
                 fcons: Optional[Callable[..., Any]] = None,
                 kwargs: Dict[Any, Any] = {},
                 omega_bounds: Tuple[float, float] = (0.1, 1.1),
                 phi_p: float = 0.5,
                 phi_g: float = 0.5,
                 max_iter: int = 100,
                 tolerance: float = 1e-6) -> Any:
        """Runs PSO algorithm.

        Parameters
        ----------
        fobj : callable
            Function to be minimized.

        lb : iterable
            The lower bounds of the solution.

        ub : iterable
            The upper bounds of the solution.

        fcons : callable
            Function for constraints that evaluates to > 0 in a successfully 
            optimized problem.

        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions.
 
        omega_bounds : tuple
            Particle velocity scaling factor lower and upper bounds.

        phi_p : float
            Scaling factor to search away from the particle's best known 
            position.

        phi_g : float
            Scaling factor to search away from the swarm's best known position.

        max_iter : int
            The maximum number of iterations for the swarm to search.

        omega_decay : bool
            Whether to implement weight decay for omega during optimization.

        tolerance : float
            Criteria for early stopping.

        Returns
        -------
        gbest_x : 1d array-like
            Swarm's best particle position.

        gbest_o : float
            Swarm's best objective function value.
        """
        pass