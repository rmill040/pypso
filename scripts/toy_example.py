import numpy as np

# Package imports
from pypso.optimizers import CPSO

def sphere(x):
    """Simple objective function.
    """
    return np.sum(x*x)


def constraint(x):
    """Simple constraint to ensure positive positions.
    """
    return x > 0


def main():
    """Runs toy examples demonstrating pypso.
    """
    kwarg_params = {
        'n_particles'  : 300,
        'n_jobs'       : 2,
        'n_dimensions' : 5,
        'verbose'      : True,
        'random_state' : 1718
    }

    # Example 1. Continuous PSO without constraint
    lb = [-5] * kwarg_params['n_dimensions']
    ub = [5]  * kwarg_params['n_dimensions']
    
    pso          = CPSO(**kwarg_params)
    x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=constraint)
    print(x_opt, o_opt)

    # Example 2. Continuous PSO with constraint


    # Example 3. Binary PSO without constraint


    # Example 4. Binary PSO with constraint



if __name__ == "__main__":
    main()