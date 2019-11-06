import numpy as np

# Package imports
from pypso.optimizers import BPSO, CPSO

def c_fobj(x):
    """Simple objective function.
    """
    return np.sum(x*x)

def b_fobj(x):
    """ADD HERE.
    """

def c_pos_constraint(x):
    """Simple constraint to ensure positive positions.
    """
    return x > 0

def c_neg_constraint(x):
    """Simple constraint to ensure positive positions.
    """
    return x < 0

def b_sum_constraint(x):
    """ADD HERE.
    """
    return np.sum(x) > 2


def main():
    """Runs toy examples demonstrating pypso.
    """
    kwarg_params = {
        'n_particles'  : 300,
        'n_jobs'       : 2,
        'n_dimensions' : 10,
        'verbose'      : True,
        'random_state' : 1718
    }

    # Example 1. Continuous PSO without constraint
    # lb = [-5] * kwarg_params['n_dimensions']
    # ub = [5]  * kwarg_params['n_dimensions']
    
    # pso          = CPSO(**kwarg_params)
    # x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=None)
    # print(x_opt, o_opt)

    # Example 2. Continuous PSO with constraint
    # lb = [-5] * kwarg_params['n_dimensions']
    # ub = [5]  * kwarg_params['n_dimensions']
    
    # pso          = CPSO(**kwarg_params)
    # x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=constraint)
    # print(x_opt, o_opt)


    # Example 3. Binary PSO without constraint
    # lb = [-1] * kwarg_params['n_dimensions']
    # ub = [0]  * kwarg_params['n_dimensions']
    
    # pso          = BPSO(**kwarg_params)
    # x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=None)
    # print(x_opt, o_opt)
    
    # Example 4. Binary PSO with constraint
    lb = [-1] * kwarg_params['n_dimensions']
    ub = [0]  * kwarg_params['n_dimensions']
    
    pso          = BPSO(**kwarg_params)
    x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub, fcons=b_sum_constraint)
    print(x_opt, o_opt)


if __name__ == "__main__":
    main()