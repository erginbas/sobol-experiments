"""Benchmark Pseudoboolean functions with known Fourier spectra.

Each factory function returns (f_callable, true_indices) where:
- f_callable(x) evaluates the function on binary vector(s) x
- true_indices is a dict mapping variable index -> true I_{i}
"""

import numpy as np


def sparse_linear(n, k, weights=None, rng=None):
    """f(x) = sum_{i in S*} w_i * x_i, where S* = {0, ..., k-1}.

    Fourier spectrum:
        f_hat(emptyset) = sum(w) / 2
        f_hat({i}) = -w_i / 2  for i in S*
        all others zero.

    Sobol indices: I_{i} = w_i^2 / 4 for i in S*, 0 otherwise.

    Parameters
    ----------
    n : int
        Total number of variables.
    k : int
        Number of influential variables (first k indices).
    weights : np.ndarray of shape (k,), optional
        Weights for the influential variables. Defaults to 1, 2, ..., k.
    rng : np.random.Generator, optional

    Returns
    -------
    f_callable : callable
        Function taking x (shape (n,) or (m, n)) -> float or (m,) array.
    true_indices : dict
        {variable_index: true I_{i}} for ALL n variables.
    """
    if weights is None:
        weights = np.arange(1, k + 1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    assert len(weights) == k

    def f(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(x[:k] @ weights)
        return x[:, :k] @ weights

    true_indices = {}
    for i in range(n):
        if i < k:
            true_indices[i] = weights[i] ** 2 / 4.0
        else:
            true_indices[i] = 0.0

    return f, true_indices


def sparse_quadratic(n, edges, weights=None):
    """f(x) = sum_{(i,j) in edges} w_{ij} * x_i * x_j.

    Fourier spectrum for a single term x_i * x_j:
        f_hat(emptyset) = 1/4, f_hat({i}) = -1/4, f_hat({j}) = -1/4,
        f_hat({i,j}) = 1/4.

    Parameters
    ----------
    n : int
    edges : list of (int, int)
    weights : np.ndarray of shape (len(edges),), optional

    Returns
    -------
    f_callable, true_indices
    """
    if weights is None:
        weights = np.ones(len(edges), dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Build Fourier coefficients using bitmasks for exact ground truth
    coeffs_dict = {}  # bitmask -> coefficient
    for idx, (i, j) in enumerate(edges):
        w = weights[idx]
        mask_empty = 0
        mask_i = 1 << i
        mask_j = 1 << j
        mask_ij = mask_i | mask_j
        for mask, val in [(mask_empty, w / 4), (mask_i, -w / 4),
                          (mask_j, -w / 4), (mask_ij, w / 4)]:
            coeffs_dict[mask] = coeffs_dict.get(mask, 0.0) + val

    def f(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            result = 0.0
            for idx, (i, j) in enumerate(edges):
                result += weights[idx] * x[i] * x[j]
            return float(result)
        result = np.zeros(x.shape[0])
        for idx, (i, j) in enumerate(edges):
            result += weights[idx] * x[:, i] * x[:, j]
        return result

    # Compute true I_{i} for each variable
    true_indices = {}
    for var in range(n):
        var_mask = 1 << var
        I_var = 0.0
        for T_mask, coeff in coeffs_dict.items():
            if T_mask != 0 and (T_mask & var_mask) != 0:
                I_var += coeff ** 2
        true_indices[var] = I_var

    return f, true_indices


def hidden_influential(n, k, order=1, rng=None):
    """k influential variables hidden among n total.

    The first k variables are influential with geometrically decreasing weights.
    order=1: f(x) = sum_i w_i * x_i (linear)
    order=2: f(x) = sum_i w_i * x_i + sum_{i<j, i,j<k} w_i*w_j/C * x_i*x_j

    Parameters
    ----------
    n : int
    k : int
    order : int, 1 or 2
    rng : np.random.Generator, optional

    Returns
    -------
    f_callable, true_indices
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Geometrically decreasing weights
    weights = np.array([2.0 ** (k - 1 - i) for i in range(k)])

    if order == 1:
        return sparse_linear(n, k, weights=weights)

    # order == 2: add pairwise interactions
    edges = []
    edge_weights = []
    C = np.sum(weights)  # normalizing constant
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
            edge_weights.append(weights[i] * weights[j] / C)

    # Build combined Fourier coefficients
    coeffs_dict = {}

    # Linear terms: f_hat(emptyset) += w_i/2, f_hat({i}) += -w_i/2
    for i in range(k):
        mask_empty = 0
        mask_i = 1 << i
        coeffs_dict[mask_empty] = coeffs_dict.get(mask_empty, 0.0) + weights[i] / 2
        coeffs_dict[mask_i] = coeffs_dict.get(mask_i, 0.0) - weights[i] / 2

    # Quadratic terms
    for idx, (i, j) in enumerate(edges):
        w = edge_weights[idx]
        mask_empty = 0
        mask_i = 1 << i
        mask_j = 1 << j
        mask_ij = mask_i | mask_j
        for mask, val in [(mask_empty, w / 4), (mask_i, -w / 4),
                          (mask_j, -w / 4), (mask_ij, w / 4)]:
            coeffs_dict[mask] = coeffs_dict.get(mask, 0.0) + val

    def f(x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        result = x[:, :k] @ weights
        for idx, (i, j) in enumerate(edges):
            result += edge_weights[idx] * x[:, i] * x[:, j]
        return float(result[0]) if single else result

    # Compute true I_{i}
    true_indices = {}
    for var in range(n):
        var_mask = 1 << var
        I_var = 0.0
        for T_mask, coeff in coeffs_dict.items():
            if T_mask != 0 and (T_mask & var_mask) != 0:
                I_var += coeff ** 2
        true_indices[var] = I_var

    return f, true_indices
