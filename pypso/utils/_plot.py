import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from typing import Any
plt.style.use('ggplot')

__all__ = ["plot_history"]


def plot_history(pso: Any) -> None:
    """Simple plot for viewing swarm's best results across iterations.
    
    Parameters
    ----------
    pso : PSO object
        Fitted PSO object.
    
    Returns
    -------
    None
    """
    n_iter: int    = len(pso.history)
    funcval: float = np.round(pso.history[-1], 4)
    plt.plot(np.arange(1, n_iter + 1, dtype=int), pso.history)

    # Handle x-axis so only integers are plotted
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Iteration")
    plt.ylabel("Objective Function")
    title: str = f"{pso.__str__()} algorithm history for {n_iter} " + \
                 f"iterations\nParticles = {pso.n_particles}, Dimensions = " + \
                 f"{pso.n_dimensions}\nOptimized function value = {funcval}"
    plt.title(title)
    plt.tight_layout()
    plt.show()