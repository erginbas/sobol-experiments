"""Tests for the Walsh-Hadamard transform and Fourier utilities."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pseudoboolean import (
    walsh_hadamard_transform,
    inverse_walsh_hadamard,
    evaluate_fourier,
    compute_I_S_exact,
    compute_variance_exact,
)


class TestWalshHadamard:
    def test_constant_function(self):
        """WHT of constant function: only f_hat(emptyset) is nonzero."""
        n = 3
        N = 1 << n
        f_values = np.ones(N) * 5.0
        coeffs = walsh_hadamard_transform(f_values)
        assert abs(coeffs[0] - 5.0) < 1e-12
        assert np.max(np.abs(coeffs[1:])) < 1e-12

    def test_single_parity(self):
        """WHT of chi_S: only f_hat(S) = 1."""
        n = 3
        N = 1 << n
        S_mask = 0b101  # S = {0, 2}
        f_values = np.array([
            (-1) ** (bin(x & S_mask).count('1')) for x in range(N)
        ], dtype=float)
        coeffs = walsh_hadamard_transform(f_values)
        for T in range(N):
            if T == S_mask:
                assert abs(coeffs[T] - 1.0) < 1e-12
            else:
                assert abs(coeffs[T]) < 1e-12

    def test_roundtrip(self):
        """Transform then inverse recovers original."""
        n = 4
        N = 1 << n
        rng = np.random.default_rng(123)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)
        recovered = inverse_walsh_hadamard(coeffs)
        np.testing.assert_allclose(recovered, f_values, atol=1e-10)

    def test_parseval(self):
        """sum f_hat(S)^2 = E[f^2] = (1/2^n) sum f(x)^2."""
        n = 4
        N = 1 << n
        rng = np.random.default_rng(456)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)
        lhs = np.sum(coeffs ** 2)
        rhs = np.mean(f_values ** 2)
        assert abs(lhs - rhs) < 1e-10

    def test_known_n3_example(self):
        """Manual check on n=3: f(x) = x_0 + 2*x_1."""
        n = 3
        N = 1 << n
        # f(x) = x_0 + 2*x_1
        f_values = np.array([
            (x & 1) + 2 * ((x >> 1) & 1) for x in range(N)
        ], dtype=float)
        coeffs = walsh_hadamard_transform(f_values)

        # f_hat(emptyset) = E[f] = E[x_0] + 2*E[x_1] = 0.5 + 1.0 = 1.5
        assert abs(coeffs[0] - 1.5) < 1e-12

        # f_hat({0}) = E[f * chi_{0}] = E[(x_0 + 2*x_1)*(-1)^{x_0}]
        # = E[x_0*(-1)^{x_0}] + 2*E[x_1*(-1)^{x_0}]
        # = E[x_0*(-1)^{x_0}] + 2*E[x_1]*E[(-1)^{x_0}]
        # = (0*1 + 1*(-1))/2 + 0 = -1/2
        assert abs(coeffs[0b001] - (-0.5)) < 1e-12

        # f_hat({1}) = -2/2 = -1.0
        assert abs(coeffs[0b010] - (-1.0)) < 1e-12

        # All other coefficients should be 0
        for T in range(N):
            if T not in [0, 0b001, 0b010]:
                assert abs(coeffs[T]) < 1e-12


class TestEvaluateFourier:
    def test_single_vector(self):
        """Evaluate at a single point."""
        n = 3
        N = 1 << n
        rng = np.random.default_rng(789)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)

        x = np.array([1, 0, 1])
        x_int = 1 + 4  # = 5
        expected = f_values[x_int]
        result = evaluate_fourier(coeffs, x)
        assert abs(result - expected) < 1e-10

    def test_batch(self):
        """Evaluate at multiple points."""
        n = 3
        N = 1 << n
        rng = np.random.default_rng(101)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)

        # All 8 inputs
        xs = np.array([
            [(i >> j) & 1 for j in range(n)] for i in range(N)
        ])
        results = evaluate_fourier(coeffs, xs)
        np.testing.assert_allclose(results, f_values, atol=1e-10)


class TestSobolIndex:
    def test_linear_function(self):
        """I_{i} for f(x) = w_0*x_0 + w_1*x_1 should be w_i^2/4."""
        n = 3
        N = 1 << n
        w = np.array([3.0, 5.0])
        f_values = np.array([
            w[0] * ((x >> 0) & 1) + w[1] * ((x >> 1) & 1)
            for x in range(N)
        ], dtype=float)
        coeffs = walsh_hadamard_transform(f_values)

        # I_{0}: all T that include bit 0
        I_0 = compute_I_S_exact(coeffs, 0b001)
        assert abs(I_0 - w[0] ** 2 / 4) < 1e-10

        # I_{1}
        I_1 = compute_I_S_exact(coeffs, 0b010)
        assert abs(I_1 - w[1] ** 2 / 4) < 1e-10

        # I_{2} (dummy variable)
        I_2 = compute_I_S_exact(coeffs, 0b100)
        assert abs(I_2) < 1e-10

    def test_variance(self):
        """Var(f) = sum of all non-empty f_hat(S)^2."""
        n = 4
        N = 1 << n
        rng = np.random.default_rng(202)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)

        var_fourier = compute_variance_exact(coeffs)
        var_direct = np.var(f_values)  # population variance
        assert abs(var_fourier - var_direct) < 1e-10

    def test_monotonicity(self):
        """I_A <= I_B when A subset B."""
        n = 4
        N = 1 << n
        rng = np.random.default_rng(303)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)

        # {0} subset {0, 1}
        I_0 = compute_I_S_exact(coeffs, 0b0001)
        I_01 = compute_I_S_exact(coeffs, 0b0011)
        assert I_0 <= I_01 + 1e-12

    def test_I_full_set_equals_variance(self):
        """I_{[n]} = Var(f)."""
        n = 4
        N = 1 << n
        rng = np.random.default_rng(404)
        f_values = rng.normal(size=N)
        coeffs = walsh_hadamard_transform(f_values)

        I_all = compute_I_S_exact(coeffs, (1 << n) - 1)
        var = compute_variance_exact(coeffs)
        assert abs(I_all - var) < 1e-10
