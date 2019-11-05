from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import cpu_count
import numpy as np
from numpy.random import Generator, PCG64
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional, 
                    Tuple, Union)

# Package imports
from ..utils.wrappers import Wrappers


__all__ = ['BasePSO']


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
                          lb: Optional[Iterable[float]] = None,
                          ub: Optional[Iterable[float]] = None,
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
            ADD HERE
        """
        # Basic error checking
        self.lb: np.ndarray
        self.ub: np.ndarray
        if lb is None or ub is None:
            self.lb = np.repeat(-np.inf, self.n_dimensions)
            self.ub = np.repeat(np.inf, self.n_dimensions)
        else:
            self.lb = np.array(lb)
            self.ub = np.array(ub)
            assert len(self.lb) == len(self.ub), \
                f"Lower- and upper-bounds must be the same length, " + \
                f"got lb = {len(self.lb)} != ub = {len(self.ub)}"
            
            assert np.all(self.ub > self.lb), \
            "All upper-bound values must be > lower-bound values"

        assert hasattr(fobj, "__call__"), \
            "Invalid function handle for fobj parameter"

        if fcons:
            assert hasattr(fcons, "__call__"), \
                "Invalid function handle for fcons parameter"

        # Bound velocities
        self.v_bounds: Tuple[float, float] = \
            (self.lb - self.ub, self.ub - self.lb)

        # Curry objective and constraint functions
        T          = Callable[[Iterable[float]], float]
        c_fobj: T  = partial(Wrappers.fobj_wrapper, fobj, kwargs)
        func: T    = Wrappers.fcons_none_wrapper if not fcons else \
                        partial(Wrappers.fcons_wrapper, fcons, kwargs)
        c_fcons: T = partial(Wrappers.is_feasible_wrapper, func)

        # Initialize particle's positions, velocities, and evaluate
        pts: np.ndarray = \
            self.rg.uniform(size=(self.n_particles, self.n_dimensions))
        x: np.ndarray   = self.lb + pts * (self.ub - self.lb)
        v: np.ndarray   = self.v_bounds[0] + \
                            pts * (self.v_bounds[1] - self.v_bounds[0])
        
        o: np.ndarray = np.array(list(map(c_fobj, x)))
        f: np.ndarray = np.array(list(map(c_fcons, x)))

        # Initialize results for each particle's best results and swarm's best
        # results
        pbest_x: np.ndarray = np.zeros_like(x)
        pbest_o: np.ndarray = np.ones(self.n_particles) * np.inf

        gbest_x: np.ndarray = np.zeros_like(x)
        gbest_o: float      = np.inf

        return {
            'x'       : x,
            'v'       : v,
            'pbest_x' : pbest_x,
            'pbest_o' : pbest_o,
            'gbest_x' : gbest_x,
            'gbest_o' : gbest_o,
            'c_fobj'  : c_fobj,
            'c_fcons' : c_fcons
        }

    @abstractmethod
    def optimize(self, 
                 fobj: Callable[..., float],
                 lb: Optional[Iterable[float]] = None,
                 ub: Optional[Iterable[float]] = None,
                 fcons: Optional[Callable[..., Any]] = None,
                 kwargs: Dict[Any, Any] = {},
                 omega: float = 0.5,
                 phi_p: float = 0.5,
                 phi_g: float = 0.5,
                 max_iter: int = 100,
                 weight_decay: bool = True,
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
 
        omega : float
            Particle velocity scaling factor.

        phi_p : float
            Scaling factor to search away from the particle's best known 
            position.

        phi_g : float
            Scaling factor to search away from the swarm's best known position.

        max_iter : int
            The maximum number of iterations for the swarm to search.

        weight_decay : bool
            Whether to implement weight decay during optimization.

        tolerance : float
            Criteria for early stopping.

        Returns
        -------
        TODO: ADD THIS!
        """
        pass