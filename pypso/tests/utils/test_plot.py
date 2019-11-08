import matplotlib.pyplot as plt
import numpy as np

from pypso.optimizers import CPSO
from pypso.utils import plot_history

def test_plot_history():
    """Tests plot_history.
    """
    # Simple objective function
    def sphere(x): return np.sum(x*x)

    # Define CPSO algorithm and run
    pso = CPSO(n_particles=5, n_dimensions=2, random_state=1718)
    lb  = [-3]*2
    ub  = [3]*2

    # Optimize and plot
    x_opt, o_opt = pso.optimize(fobj=sphere, lb=lb, ub=ub)
    try:
        plot_history(pso)
    except Exception as e:
        raise ValueError(f"failed to plot results because {e}")