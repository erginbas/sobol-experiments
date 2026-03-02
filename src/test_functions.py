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


def exponential_sparse(n, num_top=100, num_tail=5000, max_degree=4,
                       top_scale=1.0, tail_scale=0.01, rng=None):
    """Pseudoboolean function with exponentially distributed Fourier coefficients.

    Constructs a function f: {0,1}^n -> R as a sum of Walsh characters:
        f(x) = f_hat(emptyset) + sum_T c_T * chi_T(x)
    where:
    - num_top coefficients have magnitudes ~ Exp(top_scale), placed at random
      degree-<=max_degree subsets (the "signal")
    - num_tail coefficients have magnitudes ~ Exp(tail_scale), placed at random
      degree-<=max_degree subsets (the "noise floor")
    - All coefficients have random signs

    Parameters
    ----------
    n : int
        Total number of variables.
    num_top : int
        Number of large Fourier coefficients.
    num_tail : int
        Number of small tail coefficients.
    max_degree : int
        Maximum degree (subset size) for Fourier coefficients.
    top_scale : float
        Scale parameter for exponential distribution of top coefficients.
    tail_scale : float
        Scale parameter for exponential distribution of tail coefficients.
    rng : np.random.Generator, optional

    Returns
    -------
    f_callable : callable
        Function taking x (shape (n,) or (m, n)) -> float or (m,) array.
    true_indices : dict
        {variable_index: true I_{i}} for all n variables.
    coeffs_list : list of (tuple, float)
        The Fourier coefficients as (indices_tuple, value) pairs, for inspection.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _generate_coeffs(num, scale):
        """Generate Fourier coefficients at random degree-<=max_degree subsets."""
        coeffs = {}  # tuple -> coefficient value
        magnitudes = rng.exponential(scale=scale, size=num)
        signs = rng.choice([-1.0, 1.0], size=num)
        for i in range(num):
            degree = rng.integers(1, max_degree + 1)  # 1..max_degree
            subset = tuple(sorted(rng.choice(n, size=degree, replace=False)))
            coeffs[subset] = coeffs.get(subset, 0.0) + signs[i] * magnitudes[i]
        return coeffs

    top_coeffs = _generate_coeffs(num_top, top_scale)
    tail_coeffs = _generate_coeffs(num_tail, tail_scale)

    # Merge: tail first, top adds on
    all_coeffs = dict(tail_coeffs)
    for key, val in top_coeffs.items():
        all_coeffs[key] = all_coeffs.get(key, 0.0) + val

    # Remove near-zero coefficients and convert to list for fast iteration
    coeffs_list = [(indices, coeff) for indices, coeff in all_coeffs.items()
                   if abs(coeff) > 1e-15]

    # Precompute index arrays for vectorized evaluation
    _indices_arrays = [np.array(indices) for indices, _ in coeffs_list]
    _coeff_values = np.array([coeff for _, coeff in coeffs_list])

    def f(x):
        x = np.asarray(x, dtype=np.int8)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        result = np.zeros(x.shape[0])
        for idx in range(len(coeffs_list)):
            # chi_T(x) = (-1)^{sum of x_i for i in T}
            parity = x[:, _indices_arrays[idx]].sum(axis=1) % 2
            result += _coeff_values[idx] * (1.0 - 2.0 * parity)

        return float(result[0]) if single else result

    # Compute true I_{i} for each variable: I_{i} = sum_{T ni i} c_T^2
    var_to_energy = {}
    for indices, coeff in coeffs_list:
        energy = coeff ** 2
        for var in indices:
            var_to_energy[var] = var_to_energy.get(var, 0.0) + energy

    true_indices = {i: var_to_energy.get(i, 0.0) for i in range(n)}

    return f, true_indices, coeffs_list
