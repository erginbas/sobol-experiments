"""Main experiments comparing BINARY-SOBOL vs naive estimation.

Experiment 1: Correctness — verify both methods find the right top-k
Experiment 2: Sample efficiency — total evaluations vs n
Experiment 3: Varying sparsity — total evaluations vs k
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.test_functions import sparse_linear, hidden_influential
from src.binary_search import binary_sobol
from src.estimators import naive_estimate_all


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def experiment_1_correctness():
    """Experiment 1: Correctness on known sparse functions."""
    print("=" * 60)
    print("EXPERIMENT 1: Correctness")
    print("=" * 60)

    n = 128
    k = 5
    weights = np.array([32.0, 16.0, 8.0, 4.0, 2.0])
    f, true_indices = sparse_linear(n, k, weights=weights)

    samples = 2000
    rng_b = np.random.default_rng(42)
    rng_n = np.random.default_rng(42)

    # Binary-SOBOL
    top_k_binary, stats_b = binary_sobol(
        f, n, k, samples_per_estimate=samples, rng=rng_b, verbose=False
    )

    # Naive
    est_naive, se_naive, evals_naive = naive_estimate_all(f, n, samples, rng_n)
    top_k_naive_idx = np.argsort(est_naive)[::-1][:k]

    print(f"\nFunction: sparse_linear(n={n}, k={k})")
    print(f"True top-{k} variables: {list(range(k))}")
    print(f"True I values: {[true_indices[i] for i in range(k)]}")
    print()

    print("Binary-SOBOL results:")
    found_b = {v for v, _ in top_k_binary}
    for var, est in top_k_binary:
        true_val = true_indices[var]
        print(f"  var={var:3d}  est={est:8.4f}  true={true_val:8.4f}")
    print(f"  Correct: {found_b == set(range(k))}")
    print(f"  Evaluations: {stats_b.total_evaluations:,}")
    print(f"  Groups estimated: {stats_b.groups_estimated}")
    print(f"  Groups pruned: {stats_b.groups_pruned}")

    print("\nNaive results:")
    found_n = set(top_k_naive_idx.tolist())
    for var in top_k_naive_idx:
        true_val = true_indices[var]
        print(f"  var={var:3d}  est={est_naive[var]:8.4f}  true={true_val:8.4f}")
    print(f"  Correct: {found_n == set(range(k))}")
    print(f"  Evaluations: {evals_naive:,}")

    print(f"\nSpeedup: {evals_naive / stats_b.total_evaluations:.1f}x")

    # Bar chart of true vs estimated indices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binary-SOBOL
    ax = axes[0]
    vars_b = [v for v, _ in top_k_binary]
    ests_b = [e for _, e in top_k_binary]
    trues_b = [true_indices[v] for v in vars_b]
    x_pos = range(len(vars_b))
    ax.bar([p - 0.15 for p in x_pos], trues_b, 0.3, label='True', alpha=0.8)
    ax.bar([p + 0.15 for p in x_pos], ests_b, 0.3, label='Estimated', alpha=0.8)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f'var {v}' for v in vars_b])
    ax.set_ylabel('Sobol Index I_{i}')
    ax.set_title(f'Binary-SOBOL ({stats_b.total_evaluations:,} evals)')
    ax.legend()

    # Naive
    ax = axes[1]
    vars_n = top_k_naive_idx.tolist()
    ests_n = [est_naive[v] for v in vars_n]
    trues_n = [true_indices[v] for v in vars_n]
    x_pos = range(len(vars_n))
    ax.bar([p - 0.15 for p in x_pos], trues_n, 0.3, label='True', alpha=0.8)
    ax.bar([p + 0.15 for p in x_pos], ests_n, 0.3, label='Estimated', alpha=0.8)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f'var {v}' for v in vars_n])
    ax.set_ylabel('Sobol Index I_{i}')
    ax.set_title(f'Naive ({evals_naive:,} evals)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experiment1_correctness.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/results/experiment1_correctness.png")


def experiment_2_sample_efficiency():
    """Experiment 2: Sample complexity vs n (k fixed)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Sample Efficiency (varying n, k=5 fixed)")
    print("=" * 60)

    k = 5
    weights = np.array([32.0, 16.0, 8.0, 4.0, 2.0])
    samples = 1000
    n_values = [32, 64, 128, 256, 512, 1024]
    num_trials = 3

    binary_evals = []
    naive_evals = []
    binary_evals_std = []

    for n in n_values:
        b_trials = []
        n_evals = 2 * n * samples  # naive is deterministic in eval count

        for trial in range(num_trials):
            f, _ = sparse_linear(n, k, weights=weights)
            rng = np.random.default_rng(trial * 1000 + n)
            _, stats = binary_sobol(f, n, k, samples_per_estimate=samples, rng=rng)
            b_trials.append(stats.total_evaluations)

        binary_evals.append(np.mean(b_trials))
        binary_evals_std.append(np.std(b_trials))
        naive_evals.append(n_evals)

        print(f"  n={n:5d}: Binary={np.mean(b_trials):10,.0f} "
              f"+/- {np.std(b_trials):8,.0f}  "
              f"Naive={n_evals:10,d}  "
              f"Speedup={n_evals / np.mean(b_trials):.1f}x")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute evaluations
    ax = axes[0]
    ax.errorbar(n_values, binary_evals, yerr=binary_evals_std,
                marker='o', label='Binary-SOBOL', capsize=5)
    ax.plot(n_values, naive_evals, 's-', label='Naive')
    # Theoretical O(k * log(n/k) * 2 * samples)
    theoretical = [2 * k * np.log2(n / k) * 2 * samples for n in n_values]
    ax.plot(n_values, theoretical, 'k--', alpha=0.5, label='O(k log(n/k))')
    ax.set_xlabel('Number of variables n')
    ax.set_ylabel('Total function evaluations')
    ax.set_title(f'Sample Complexity (k={k})')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Speedup
    ax = axes[1]
    speedups = [n_e / b_e for n_e, b_e in zip(naive_evals, binary_evals)]
    ax.plot(n_values, speedups, 'ro-', markersize=8)
    ax.set_xlabel('Number of variables n')
    ax.set_ylabel('Speedup (Naive / Binary)')
    ax.set_title(f'Speedup Factor (k={k})')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experiment2_efficiency.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/results/experiment2_efficiency.png")


def experiment_3_varying_sparsity():
    """Experiment 3: Evaluations vs k (n fixed)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Varying Sparsity (n=256 fixed, varying k)")
    print("=" * 60)

    n = 256
    samples = 1000
    k_values = [1, 2, 5, 10, 25, 50]

    binary_evals = []
    naive_evals = []

    for k in k_values:
        # Create weights: geometrically decreasing
        weights = np.array([2.0 ** (k - 1 - i) for i in range(k)])
        f, _ = sparse_linear(n, k, weights=weights)

        rng = np.random.default_rng(42)
        _, stats = binary_sobol(f, n, k, samples_per_estimate=samples, rng=rng)

        n_evals = 2 * n * samples

        binary_evals.append(stats.total_evaluations)
        naive_evals.append(n_evals)

        print(f"  k={k:3d}: Binary={stats.total_evaluations:10,d}  "
              f"Naive={n_evals:10,d}  "
              f"Speedup={n_evals / stats.total_evaluations:.1f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, binary_evals, 'o-', label='Binary-SOBOL', markersize=8)
    ax.plot(k_values, naive_evals, 's-', label='Naive', markersize=8)
    ax.set_xlabel('Number of influential variables k')
    ax.set_ylabel('Total function evaluations')
    ax.set_title(f'Effect of Sparsity (n={n})')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experiment3_sparsity.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to experiments/results/experiment3_sparsity.png")


if __name__ == '__main__':
    experiment_1_correctness()
    experiment_2_sample_efficiency()
    experiment_3_varying_sparsity()
    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)
