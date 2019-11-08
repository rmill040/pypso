import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Package imports
from pypso.optimizers import BPSO, CPSO
from pypso.utils import plot_history


# Globally load data from sklearn
X, y = load_breast_cancer(return_X_y=True)
X    = (X - X.mean()) / X.std()

# Define linear model for comparison with PSO
clf = LogisticRegression(solver="lbfgs")
clf.fit(X, y)
auc_sk = np.round(roc_auc_score(y, clf.predict_proba(X)[:, -1]), 4)

# Define objective functions
def fobj_lr(w):
    """Optimize logistic regression weights using AUC metric.

    Parameters
    ----------
    w : 1d array-like
        Weights for logistic regression.

    Returns
    -------
    float
        AUC metric.
    """
    # Linear combo and create probabilities
    z = w[0] + np.sum(w[1:]*X, axis=1)
    p = 1 / (1 + np.exp(-z))

    # Want to minimize objective so take 1 - AUC
    return 1 - roc_auc_score(y, p)


def fobj_lr_constraint(w):
    """Set arbitrary constraint for logistic regression weights.
    
    Parameters
    ----------
    w : 1d array-like
        Weights for logistic regression.
    
    Returns
    -------
    1d array-like
        Boolean vector indicating if element satisfies condition.
    """
    return abs(w) < 2


def fobj_fs(b):
    """Optimize feature selection using AUC metric.
    
    Parameters
    ----------
    b : 1d array-like
        Array indicating if feature is selected (1=yes, 0=no).
    
    Returns
    -------
    float
        AUC metric.
    """
    b = b.astype(bool)
    try:
        clf = LogisticRegression(solver="lbfgs").fit(X[:, b], y)
        return 1 - roc_auc_score(y, clf.predict_proba(X[:, b])[:, -1])
    except Exception as e:
        return np.inf

def fobj_fs_constraint(b):
    """Set arbitrary constraint for feature selection to use 50% or less of the 
    total feature space.
    
    Parameters
    ----------
    b : 1d array-like
        Array indicating if feature is selected (1=yes, 0=no).
    
    Returns
    -------
    1d array-like
        Boolean vector indicating if element satisfies condition.
    """
    return np.sum(b) <= 10


def main():
    """Runs toy examples demonstrating pypso.
    """
    # Kwargs for PSO algorithms
    kwarg_params = {
        'n_particles'  : 300,
        'n_jobs'       : 4,
        'verbose'      : True,
        'random_state' : 1718
    }

    #######################
    # CONTINUOUS EXAMPLES #
    #######################

    n_dimensions = X.shape[1] + 1 # +1 for bias term
    lb           = [-4] * n_dimensions
    ub           = [4]  * n_dimensions

    """Example 1. Continuous PSO without constraint"""

    print(f"{'-'*50}\nEXAMPLE 1 - CONTINUOUS PSO WITHOUT CONSTRAINT\n{'-'*50}")

    pso          = CPSO(n_dimensions=n_dimensions, **kwarg_params)
    w_opt, o_opt = pso.optimize(fobj=fobj_lr, lb=lb, ub=ub, fcons=None)
    plot_history(pso)

    # Print solution
    names   = [f"f{i}" for i in range(1, X.shape[1] + 1)]
    weights = np.round(w_opt, 2).tolist()
    sol     = f"{weights[0]} + "
    for i in range(X.shape[1]):
        if (i + 1) % 5 == 0: sol += "\n"
        sol += f"{weights[i+1]}*{names[i]}"
        if i < X.shape[1]-1: sol += " + "
    print(f"\nLinear Solution:\n{sol}")

    # Sanity check
    print("\nSanity check:")
    status = np.all((np.array(lb) < w_opt) & (w_opt < np.array(ub)))
    print(f"\tall weights within bounds? {status}\n")

    # Compare to sklearn logistic regression
    auc_pso = np.round(1 - o_opt, 4)
    print("Comparison to sklearn:")
    print(f"\tsklearn logistic regression AUC = {auc_sk}")
    print(f"\tPSO logistic regression AUC     = {auc_pso}\n")

    """Example 2. Continuous PSO with constraint"""

    print(f"{'-'*50}\nEXAMPLE 2 - CONTINUOUS PSO WITH CONSTRAINT\n{'-'*50}")

    pso          = CPSO(n_dimensions=n_dimensions, **kwarg_params)
    w_opt, o_opt = pso.optimize(fobj=fobj_lr, lb=lb, ub=ub, fcons=fobj_lr_constraint)
    plot_history(pso)

    # Print solution
    names   = [f"f{i}" for i in range(1, X.shape[1] + 1)]
    weights = np.round(w_opt, 2).tolist()
    sol     = f"{weights[0]} + "
    for i in range(X.shape[1]):
        if (i + 1) % 5 == 0: sol += "\n"
        sol += f"{weights[i+1]}*{names[i]}"
        if i < X.shape[1]-1: sol += " + "
    print(f"\nLinear Solution:\n{sol}")

    print("\nSanity check:")
    status = np.all((np.array(lb) < w_opt) & (w_opt < np.array(ub)))
    print(f"\tall weights within bounds? {status}")
    
    status = np.all(fobj_lr_constraint(w_opt))
    print(f"\tall weights satisfy constraint? {status}\n")

    # Compare to sklearn logistic regression
    auc_pso = np.round(1 - o_opt, 4)
    print("Comparison to sklearn:")
    print(f"\tsklearn logistic regression AUC = {auc_sk}")
    print(f"\tPSO logistic regression AUC     = {auc_pso}\n")
    
    ###################
    # BINARY EXAMPLES #
    ###################

    n_dimensions = X.shape[1]
    lb           = [0] * X.shape[1]
    ub           = [1] * X.shape[1]

    """Example 3. Binary PSO without constraint"""

    print(f"{'-'*50}\nEXAMPLE 3 - BINARY PSO WITHOUT CONSTRAINT\n{'-'*50}")

    pso          = BPSO(n_dimensions=n_dimensions, **kwarg_params)
    b_opt, o_opt = pso.optimize(fobj=fobj_fs, 
                                lb=lb, 
                                ub=ub, 
                                fcons=None, 
                                max_iter=5)
    plot_history(pso)
    
    # Features selected
    print("\nFeatures selected:")
    print(f"\tn   = {int(np.sum(b_opt))}")
    print(f"\tids = {np.where(b_opt)[0]}")

    # Compare to performance without feature selection
    auc_pso = np.round(1 - o_opt, 4)
    print("\nComparison to no feature selection:")
    print(f"\tno feature selection AUC = {auc_sk}")
    print(f"\tfeature selection AUC    = {auc_pso}\n")

    """Example 4. Binary PSO with constraint"""
    
    print(f"{'-'*50}\nEXAMPLE 4 - BINARY PSO WITH CONSTRAINT\n{'-'*50}")

    pso          = BPSO(n_dimensions=n_dimensions, **kwarg_params)
    b_opt, o_opt = pso.optimize(fobj=fobj_fs, 
                                lb=lb, 
                                ub=ub, 
                                fcons=fobj_fs_constraint, 
                                max_iter=5)
    plot_history(pso)
    
    # Features selected
    print("\nFeatures selected:")
    print(f"\tn   = {int(np.sum(b_opt))}")
    print(f"\tids = {np.where(b_opt)[0]}")

    # Sanity check
    print("\nSanity check:")
    status = np.all(fobj_fs_constraint(b_opt))
    print(f"\tall weights satify constraint? {status}")

    # Compare to performance without feature selection
    auc_pso = np.round(1 - o_opt, 4)
    print("\nComparison to no feature selection:")
    print(f"\tno feature selection AUC = {auc_sk}")
    print(f"\tfeature selection AUC    = {auc_pso}")

if __name__ == "__main__":
    main()