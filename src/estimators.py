"""Pick-freeze Monte Carlo estimator for Sobol indices I_S.

The key identity: for x ~ Uniform({0,1}^n) and z = (x'_S, x_{bar{S}})
where x' is independent of x:

    I_S = E[f(x)(f(x) - f(z))]

So f(x)(f(x) - f(z)) is an unbiased estimator of I_S.
"""

import numpy as np


def pick_freeze_estimate(f_callable, n, S_indices, num_samples, rng=None):
    """Estimate I_S using the pick-freeze method.

    Parameters
    ----------
    f_callable : callable
        Function taking binary array of shape (m, n) -> array of shape (m,).
    n : int
        Number of variables.
    S_indices : list of int
        Variable indices in the subset S.
    num_samples : int
        Number of sample pairs.
    rng : np.random.Generator, optional

    Returns
    -------
    estimate : float
        Point estimate of I_S.
    std_error : float
        Standard error of the estimate.
    num_evals : int
        Number of function evaluations used (= 2 * num_samples).
    """
    if rng is None:
        rng = np.random.default_rng()

    S_indices = list(S_indices)

    # Generate independent samples
    x = rng.integers(0, 2, size=(num_samples, n))
    x_prime = rng.integers(0, 2, size=(num_samples, n))

    # Construct z: agrees with x outside S, resampled from x' within S
    z = x.copy()
    if len(S_indices) > 0:
        z[:, S_indices] = x_prime[:, S_indices]

    # Evaluate f on both
    fx = np.asarray(f_callable(x), dtype=float)
    fz = np.asarray(f_callable(z), dtype=float)

    # Unbiased estimator: f(x) * (f(x) - f(z))
    contributions = fx * (fx - fz)

    estimate = float(np.mean(contributions))
    std_error = float(np.std(contributions, ddof=1) / np.sqrt(num_samples))

    return estimate, std_error, 2 * num_samples


def naive_estimate_all(f_callable, n, num_samples, rng=None):
    """Estimate I_{i} for every variable i independently.

    Baseline method: runs n separate pick-freeze estimations.

    Parameters
    ----------
    f_callable : callable
    n : int
    num_samples : int
        Samples per variable.
    rng : np.random.Generator, optional

    Returns
    -------
    estimates : np.ndarray of shape (n,)
        Estimated I_{i} for each variable.
    std_errors : np.ndarray of shape (n,)
    total_evals : int
    """
    if rng is None:
        rng = np.random.default_rng()

    estimates = np.zeros(n)
    std_errors = np.zeros(n)
    total_evals = 0

    for i in range(n):
        est, se, ne = pick_freeze_estimate(f_callable, n, [i], num_samples, rng)
        estimates[i] = est
        std_errors[i] = se
        total_evals += ne

    return estimates, std_errors, total_evals
