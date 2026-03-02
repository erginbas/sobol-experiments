"""Pseudoboolean function framework and Fourier analysis on {0,1}^n.

A Pseudoboolean function f: {0,1}^n -> R has a unique Fourier expansion:
    f(x) = sum_{S subset [n]} f_hat(S) * chi_S(x)
where chi_S(x) = (-1)^{sum_{i in S} x_i} are the Walsh characters.

Subsets S are represented as integer bitmasks: bit i is set iff i in S.
"""

import numpy as np


def walsh_hadamard_transform(f_values):
    """Compute all 2^n Fourier coefficients from the truth table.

    Uses the fast in-place butterfly algorithm, O(n * 2^n).

    Parameters
    ----------
    f_values : np.ndarray of shape (2^n,)
        Function values indexed by the integer encoding of x.

    Returns
    -------
    np.ndarray of shape (2^n,)
        Fourier coefficients, where index i encodes subset S as a bitmask.
    """
    a = f_values.astype(float).copy()
    N = len(a)
    n = int(np.log2(N))
    assert N == 1 << n, "Length must be a power of 2"

    for i in range(n):
        step = 1 << i
        for j in range(N):
            if j & step == 0:
                u = a[j]
                v = a[j | step]
                a[j] = u + v
                a[j | step] = u - v

    a /= N
    return a


def inverse_walsh_hadamard(coeffs):
    """Reconstruct truth table from Fourier coefficients.

    Parameters
    ----------
    coeffs : np.ndarray of shape (2^n,)
        Fourier coefficients indexed by bitmask.

    Returns
    -------
    np.ndarray of shape (2^n,)
        Function values f(x) for each x.
    """
    a = coeffs.astype(float).copy()
    N = len(a)
    n = int(np.log2(N))
    assert N == 1 << n

    for i in range(n):
        step = 1 << i
        for j in range(N):
            if j & step == 0:
                u = a[j]
                v = a[j | step]
                a[j] = u + v
                a[j | step] = u - v

    return a


def evaluate_fourier(coeffs, x):
    """Evaluate f(x) = sum_S f_hat(S) * chi_S(x) from Fourier coefficients.

    Parameters
    ----------
    coeffs : np.ndarray of shape (2^n,)
        Fourier coefficients indexed by bitmask.
    x : np.ndarray of shape (n,) or (m, n)
        Binary input vector(s).

    Returns
    -------
    float or np.ndarray of shape (m,)
    """
    x = np.asarray(x)
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    n = x.shape[1]
    N = 1 << n
    assert len(coeffs) == N

    result = np.zeros(x.shape[0])
    for S_mask in range(N):
        if coeffs[S_mask] == 0:
            continue
        # chi_S(x) = (-1)^{sum_{i in S} x_i}
        bits_in_S = np.array([
            (S_mask >> i) & 1 for i in range(n)
        ], dtype=x.dtype)
        parity = x @ bits_in_S  # sum of x_i for i in S
        chi = (-1.0) ** parity
        result += coeffs[S_mask] * chi

    return result[0] if single else result


def compute_I_S_exact(coeffs, S_mask):
    """Compute I_S = sum_{T: T & S_mask != 0, T != 0} f_hat(T)^2 exactly.

    Parameters
    ----------
    coeffs : np.ndarray of shape (2^n,)
    S_mask : int
        Bitmask for the subset S.

    Returns
    -------
    float
    """
    N = len(coeffs)
    total = 0.0
    for T in range(1, N):  # skip T=0 (empty set)
        if T & S_mask != 0:
            total += coeffs[T] ** 2
    return total


def compute_variance_exact(coeffs):
    """Compute Var(f) = sum_{T != 0} f_hat(T)^2.

    Parameters
    ----------
    coeffs : np.ndarray of shape (2^n,)

    Returns
    -------
    float
    """
    return float(np.sum(coeffs[1:] ** 2))
