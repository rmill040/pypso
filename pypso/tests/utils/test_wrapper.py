import numpy as np
from pypso.utils import Wrappers

def test_fobj_wrapper():
    """Tests Wrappers.fobj_wrapper.
    """
    # Define simple function
    def func(x): return list(map(lambda z: z*z, x))
    
    # Compare results
    x         = [1, 2, 3]
    f_results = func(x)
    w_results = Wrappers.fobj_wrapper(func, {}, x)
    assert(f_results == w_results)

def test_is_feasible_wrapper():
    """Tests Wrappers.is_feasible_wrapper.
    """
    # Define constraint function
    def cons(x): return x > 0

    # Feasible solution
    x     = np.array([1, 1, 1])
    f_sol = Wrappers.is_feasible_wrapper(cons, x)
    assert(f_sol)

    # Infeasible solution
    x *= -1
    i_sol = Wrappers.is_feasible_wrapper(cons, x)
    assert(i_sol == False)

def test_is_feasible_wrapper():
    """Tests Wrappers.is_feasible_wrapper.
    """
    # Define constraint function
    def cons(x): return x > 0

    # Feasible solution
    x     = np.array([1, 1, 1])
    f_sol = Wrappers.is_feasible_wrapper(cons, x)
    assert(f_sol)

    # Infeasible solution
    x *= -1
    i_sol = Wrappers.is_feasible_wrapper(cons, x)
    assert(i_sol == False)

def test_fcons_none_wrapper():
    """Tests Wrappers.fcons_none_wrapper.
    """
    x = [1, 2, 3]
    assert(Wrappers.fcons_none_wrapper(x))

def test_fcons_wrapper():
    """Tests Wrappers.fcons_wrapper.
    """
    # Define constraint function
    def cons(x): return x > 0
    
    x    = np.array([-1, 1, -1])
    true = np.array([False, True, False])
    sol  = Wrappers.fcons_wrapper(cons, {}, x)
    assert(np.all(sol == true))