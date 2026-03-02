"""Tests for the BINARY-SOBOL algorithm."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.binary_search import binary_sobol
from src.test_functions import sparse_linear, hidden_influential
from src.estimators import naive_estimate_all


class TestBinarySobol:
    def test_finds_top_k_linear(self):
        """Correctly identifies top-k variables in a linear function."""
        n = 16
        k = 3
        weights = np.array([16.0, 8.0, 2.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(42)
        top_k, stats = binary_sobol(f, n, k, samples_per_estimate=2000, rng=rng)

        found_vars = {var for var, est in top_k}
        assert found_vars == {0, 1, 2}, f"Expected {{0,1,2}}, got {found_vars}"

    def test_finds_top_1(self):
        """k=1: finds the single most influential variable."""
        n = 32
        k = 1
        weights = np.array([10.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(123)
        top_k, stats = binary_sobol(f, n, k, samples_per_estimate=2000, rng=rng)

        assert top_k[0][0] == 0

    def test_pruning_reduces_evaluations(self):
        """Binary search uses fewer evaluations than naive for sparse problems."""
        n = 64
        k = 3
        weights = np.array([10.0, 5.0, 2.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        samples = 1000

        _, stats_binary = binary_sobol(f, n, k,
                                       samples_per_estimate=samples, rng=rng1)

        _, _, naive_evals = naive_estimate_all(f, n, samples, rng2)

        assert stats_binary.total_evaluations < naive_evals, \
            f"Binary ({stats_binary.total_evaluations}) should use fewer " \
            f"evals than naive ({naive_evals})"

    def test_pruning_happens(self):
        """Pruning should occur for sparse problems."""
        n = 32
        k = 2
        weights = np.array([10.0, 5.0])
        f, true_indices = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(456)
        _, stats = binary_sobol(f, n, k, samples_per_estimate=2000, rng=rng)

        assert stats.groups_pruned > 0, "Expected some groups to be pruned"

    def test_hidden_influential(self):
        """Finds influential variables in the hidden_influential benchmark."""
        n = 32
        k = 3
        f, true_indices = hidden_influential(n, k, order=1)

        rng = np.random.default_rng(789)
        top_k, stats = binary_sobol(f, n, k, samples_per_estimate=2000, rng=rng)

        found_vars = {var for var, est in top_k}
        assert found_vars == {0, 1, 2}, f"Expected {{0,1,2}}, got {found_vars}"

    def test_returns_correct_count(self):
        """Always returns exactly k results."""
        n = 16
        k = 5
        f, _ = sparse_linear(n, 3)

        rng = np.random.default_rng(101)
        top_k, _ = binary_sobol(f, n, k, samples_per_estimate=1000, rng=rng)

        assert len(top_k) == k
