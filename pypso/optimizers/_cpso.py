import logging
from multiprocessing import Pool
import numpy as np
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

# Package imports
from ..base import BasePSO

__all__ = ['CPSO']
_LOGGER = logging.getLogger(__name__)


class CPSO(BasePSO):
    """Continuous particle swarm optimization algorithm.
    
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
    def __init__(self,
                 n_particles: int,
                 n_dimensions: int,
                 verbose: bool = False,
                 n_jobs: int = 1,
                 random_state: Optional[int] = None) -> None:
        super().__init__(n_particles=n_particles,
                         n_dimensions=n_dimensions,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         random_state=random_state)

    def __str__(self):
        """Returns name of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Name of class.
        """
        return "CPSO"

    def optimize(self, 
                 fobj: Callable[..., float],
                 lb: Optional[Iterable[float]] = None,
                 ub: Optional[Iterable[float]] = None,
                 fcons: Optional[Callable[..., Any]] = None,
                 kwargs: Dict[Any, Any] = {},
                 omega_bounds: Tuple[float, float] = (0.1, 0.9),
                 phi_p: float = 0.5,
                 phi_g: float = 0.5,
                 max_iter: int = 100,
                 tolerance: float = 1e-6) -> Any:
        """Runs continuous PSO algorithm.

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
            Particle velocity scaling factor lower and upper bounds. To obtain a 
            constant scaling factor, set the omega_bounds to the same number

        phi_p : float
            Scaling factor to search away from the particle's best known 
            position.

        phi_g : float
            Scaling factor to search away from the swarm's best known position.

        max_iter : int
            The maximum number of iterations for the swarm to search.

        tolerance : float
            Criteria for early stopping.

        Returns
        -------
        gbest_x : 1d array-like
            Swarm's best particle position
            
        gbest_o : float
            Swarm's best objective function value
        """
        x: np.ndarray
        v: np.ndarray

        pbest_x: np.ndarray
        pbest_o: np.ndarray

        gbest_x: np.ndarray
        gbest_o: float

        c_obj: Callable[[Iterable[float]], float]
        c_fcons: Callable[[Iterable[float]], float]
        params: Dict[str, Any]
        
        # Define function mapper
        mapper = Pool(self.n_jobs).map if self.n_jobs > 1 else \
                    lambda func, x: list(map(func, x))

        # Create swarm
        if self.verbose:
            _LOGGER.info("initializing swarm")
        params = self._initialize_swarm(fobj=fobj,
                                        lb=lb,
                                        ub=ub,
                                        fcons=fcons,
                                        kwargs=kwargs)
        x       = params.pop('x')
        v       = params.pop('v')
        pbest_x = params.pop('pbest_x')
        pbest_o = params.pop('pbest_o')
        gbest_x = params.pop('gbest_x')
        gbest_o = params.pop('gbest_o')
        c_fobj  = params.pop('c_fobj')
        c_fcons = params.pop('c_fcons')

        # Run optimization
        o: np.ndarray
        f: np.ndarray
        rp: np.ndarray
        rg: np.ndarray
        mask_lb: np.ndarray
        mask_ub: np.ndarray
        idx: np.ndarray
        
        it: int = 1
        while it <= max_iter:

            # Iteratively update omega
            omega = omega_bounds[1] - \
                it*((omega_bounds[1] - omega_bounds[0])/max_iter)
            
            # Update particles' velocities
            rp = self.rg.uniform(size=(self.n_particles, self.n_dimensions))
            rg = self.rg.uniform(size=(self.n_particles, self.n_dimensions))
            v  = omega*v + phi_p*rp*(pbest_x - x) + phi_g*rg*(gbest_x - x)
            x += v
            
            # Adjust positions based on bounds
            mask_lb = x < self.lb
            mask_ub = x > self.ub
            x       = x*(~np.logical_or(mask_lb, mask_ub)) + \
                        self.lb*mask_lb + self.ub*mask_ub

            # Update objectives and constraints
            o = np.array(mapper(c_fobj, x))
            f = np.array(mapper(c_fcons, x))

            # Update particles' best results (if constraints are satisfied)
            idx          = np.logical_and((o < pbest_o), f)
            pbest_x[idx] = x[idx].copy()
            pbest_o[idx] = o[idx]

            # Update swarm's best results (if constraints are satisfied)
            i: int = np.argmin(pbest_o)
            if pbest_o[i] < gbest_o:
                # Check stopping criteria
                ydiff = np.linalg.norm(gbest_o - pbest_o[i])
                xdiff = np.linalg.norm(gbest_x - pbest_x[i])
                ratio = ydiff/(xdiff + 1e-10)
                if ratio < tolerance: 
                    if self.verbose:
                        _LOGGER.info("optimization converged: " + \
                                     f"{it-1}/{max_iter} - stopping " + \
                                     "criteria below tolerance")
                    break
                
                # Stopping criteria not met so update gbest results
                gbest_x = pbest_x[i].copy()
                gbest_o = pbest_o[i]

                if self.verbose:
                    _LOGGER.info(f"new swarm best: {it}/{max_iter} - {gbest_o}")

            # Continue optimization
            it += 1

        # Maximum iterations reached      
        if self.verbose and it > max_iter:
            _LOGGER.warn(f"optimization did not converge in {max_iter} " + \
                         "iterations")      

        # Check if solution is feasible based on constraints
        if self.verbose and not c_fcons(gbest_x):
            _LOGGER.warn("optimization could not find a feasible solution " + \
                         "based on specified constraints")        
        
        return gbest_x, gbest_o