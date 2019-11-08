import numpy as np

from pypso.optimizers import CPSO

# Simple objective and constraint function
def sphere(x): return np.sum(x*x)
def cons(x)  : return x > 0

# Define global class
pso = CPSO(n_particles=5, n_dimensions=2, random_state=1718)
lb  = [-3]*2
ub  = [3]*2

def test_str():
    """Tests __str__ method of CPSO.
    """
    assert(pso.__str__() == "CPSO")

def test_initialize_swarm():
    """Tests _initialize_swarm method of CPSO.
    """
    # Get initial params
    params = pso._initialize_swarm(fobj=sphere, lb=lb, ub=ub, fcons=cons)

    # Random naive checks
    assert(params['x'].shape == (5, 2))
    assert(np.all(params['lb'] == np.array(lb)))
    assert(np.all(params['ub'] == np.array(ub)))

def test_optimize():
    """Tests optimize method of CPSO.
    """
    x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=cons)
    assert(np.all(x_opt > 0))