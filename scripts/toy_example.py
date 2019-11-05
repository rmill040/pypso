import numpy as np

# Package imports
from pypso.optimizers import CPSO

def sphere(x):
    return np.sum(x*x)

def cons(x):
    return x > 1

def main():
    """ADD HERE.
    """    
    lb = [-10]*5
    ub = [10]*5
    
    pso = CPSO(n_particles=200, n_dimensions=5, n_jobs=4, random_state=1718)
    pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=cons)

if __name__ == "__main__":
    main()