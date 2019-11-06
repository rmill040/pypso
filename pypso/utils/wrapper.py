import numpy as np
from typing import Any, Callable, Dict, Iterable, Tuple


class Wrappers:
    """Static class to hold function wrappers.
    """
    @staticmethod
    def fobj_wrapper(fobj: Callable[..., float], 
                     kwargs: Dict[Any, Any], 
                     x: Iterable[float]) -> float:
        """Wraps an objective function to return its function value.

        Parameters
        ----------
        fobj : callable
            Function to be minimized.

        kwargs : dict
            Additional keyword arguments to be passed to fobj.

        x : 1d array-like
            Input data.

        Returns
        -------
        float
            Evaluated objective function.
        """
        return fobj(x, **kwargs)

    @staticmethod
    def is_feasible_wrapper(func: Callable[..., Any], 
                            x: Iterable[float]) -> Any:
        """Wraps a constraint function to determine if solution is feasible 
        subject to constraints.
        
        Parameters
        ----------
        func : callable
            Function to evaluate constraints.

        x : 1d array-like
            Input data.
        
        Returns
        -------
        1d array-like
            Array with status of each constraint evaluated on provided solution.
        """
        return np.all(func(x).astype(float) > 0.0)

    @staticmethod
    def fcons_none_wrapper(x: Iterable[float]) -> Any:
        """Creates default wrapper for optimization problems with no 
        constraints.

        Parameters
        ----------
        x : 1d array-like
            Input data.
        
        Returns
        -------
        1d array-like
            Array with 1 element as 1.
        """
        return np.array([1])

    @staticmethod
    def fcons_wrapper(fcons: Callable[..., Any], 
                      kwargs: Dict[Any, Any], 
                      x: Iterable[float]) -> Any:
        """Creates wrapper for single constraint functions.
        
        Parameters
        ----------
        fcons : callable
            Function that evaluates to >= 0.0 in a successfully optimized 
            problem.

        kwargs : dict
            Additional keyword arguments to be passed to fcons.

        x : 1d array-like
            Input data.

        Returns
        -------
        1d array-like
            Array with value of each constraint based on the solution.
        """
        return np.array(fcons(x, **kwargs))