"""Tests for the pick-freeze Sobol index estimator."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.estimators import pick_freeze_estimate, naive_estimate_all
from src.test_functions import sparse_linear, sparse_quadratic


class TestPickFreeze:
    def test_unbiased_linear(self):
        """Mean estimate converges to true I_{i} for linear function."""
        n = 8
        k = 3
        weights = np.array([4.0, 2.0, 1.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(42)
        num_samples = 10000

        # Test variable 0 (influential)
        est, se, _ = pick_freeze_estimate(f, n, [0], num_samples, rng)
        true_val = true_indices[0]  # 4^2/4 = 4.0
        assert abs(est - true_val) < 3 * se, \
            f"Estimate {est} not within 3 SE of true {true_val}"

        # Test variable 5 (dummy)
        est_dummy, se_dummy, _ = pick_freeze_estimate(f, n, [5], num_samples, rng)
        assert abs(est_dummy) < 3 * se_dummy + 0.01, \
            f"Dummy estimate {est_dummy} should be near 0"

    def test_group_index(self):
        """I_S for a group includes all variables in S."""
        n = 8
        k = 3
        weights = np.array([4.0, 2.0, 1.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(123)
        num_samples = 10000

        # I_{0,1} should be I_0 + I_1 for linear (disjoint Fourier support)
        est_01, se_01, _ = pick_freeze_estimate(f, n, [0, 1], num_samples, rng)
        true_01 = true_indices[0] + true_indices[1]  # 4.0 + 1.0 = 5.0
        assert abs(est_01 - true_01) < 3 * se_01, \
            f"Group estimate {est_01} not within 3 SE of true {true_01}"

    def test_empty_set(self):
        """I_emptyset = 0: resampling nothing means f(x) = f(z)."""
        n = 4
        f, _ = sparse_linear(n, 2)
        rng = np.random.default_rng(456)
        est, se, _ = pick_freeze_estimate(f, n, [], 5000, rng)
        assert abs(est) < 0.01, f"I_emptyset should be ~0, got {est}"

    def test_full_set_equals_variance(self):
        """I_{[n]} should approximate Var(f)."""
        n = 8
        k = 3
        weights = np.array([4.0, 2.0, 1.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(789)
        num_samples = 10000

        est_all, se_all, _ = pick_freeze_estimate(
            f, n, list(range(n)), num_samples, rng
        )
        true_var = sum(true_indices.values())
        assert abs(est_all - true_var) < 3 * se_all, \
            f"I_[n] estimate {est_all} not near Var(f)={true_var}"

    def test_quadratic_function(self):
        """Estimator works for functions with pairwise interactions."""
        n = 6
        edges = [(0, 1), (2, 3)]
        weights = np.array([3.0, 1.0])
        f, true_indices = sparse_quadratic(n, edges, weights)

        rng = np.random.default_rng(101)
        num_samples = 20000

        # Variable 0 should have nonzero index (involved in edge (0,1))
        est, se, _ = pick_freeze_estimate(f, n, [0], num_samples, rng)
        assert abs(est - true_indices[0]) < 3 * se + 0.01

        # Variable 5 (not in any edge)
        est_5, se_5, _ = pick_freeze_estimate(f, n, [5], num_samples, rng)
        assert abs(est_5) < 3 * se_5 + 0.01


class TestNaiveEstimateAll:
    def test_finds_correct_ranking(self):
        """Naive method should rank variables correctly with enough samples."""
        n = 8
        k = 3
        weights = np.array([8.0, 4.0, 1.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(42)
        estimates, _, total_evals = naive_estimate_all(f, n, 2000, rng)

        # Top 3 should be variables 0, 1, 2
        top3 = np.argsort(estimates)[::-1][:3]
        assert set(top3) == {0, 1, 2}

        # Check total evals: 2 * n * num_samples
        assert total_evals == 2 * n * 2000
