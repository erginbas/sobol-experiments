"""Non-exact experiment: exponentially distributed Fourier coefficients.

Constructs a Pseudoboolean function on n=1000 variables with:
- 100 large Fourier coefficients at degree-<=4 subsets (Exp(1.0) magnitudes)
- 5000 small tail coefficients at degree-<=4 subsets (Exp(0.01) magnitudes)

Uses BINARY-SOBOL to recover the top-10 variables by Sobol index.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.test_functions import exponential_sparse
from src.binary_search import binary_sobol
from src.estimators import pick_freeze_estimate


def run_noisy_experiment():
    print("=" * 70)
    print("NON-EXACT EXPERIMENT: Exponentially Distributed Fourier Coefficients")
    print("=" * 70)

    # Parameters
    n = 1000
    num_top = 100
    num_tail = 5000
    max_degree = 4
    k = 10
    samples_per_estimate = 2000
    seed = 42

    rng_func = np.random.default_rng(seed)
    f, true_indices, coeffs_list = exponential_sparse(
        n, num_top=num_top, num_tail=num_tail, max_degree=max_degree,
        top_scale=1.0, tail_scale=0.01, rng=rng_func
    )

    # Summary statistics
    total_energy = sum(c ** 2 for _, c in coeffs_list)
    sorted_coeffs = sorted(coeffs_list, key=lambda x: x[1] ** 2, reverse=True)
    top_100_energy = sum(c ** 2 for _, c in sorted_coeffs[:100])
    tail_energy = total_energy - top_100_energy

    print(f"\nFunction construction:")
    print(f"  n = {n}")
    print(f"  Total Fourier coefficients: {len(coeffs_list)}")
    print(f"  Total energy (Var(f)): {total_energy:.4f}")
    print(f"  Top-100 coeff energy: {top_100_energy:.4f} "
          f"({100 * top_100_energy / total_energy:.1f}%)")
    print(f"  Tail energy: {tail_energy:.4f} "
          f"({100 * tail_energy / total_energy:.1f}%)")

    # True top-10 variables
    true_sorted = sorted(true_indices.items(), key=lambda x: -x[1])
    true_top_10 = true_sorted[:k]
    true_top_10_vars = {v for v, _ in true_top_10}

    print(f"\nTrue top-{k} variables by Sobol index I_{{i}}:")
    print(f"  {'Rank':>4}  {'Var':>5}  {'I_i':>10}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*10}")
    for rank, (var, I_val) in enumerate(true_top_10, 1):
        print(f"  {rank:4d}  {var:5d}  {I_val:10.6f}")

    # Count how many variables have nonzero Sobol index
    nonzero_vars = sum(1 for v in true_indices.values() if v > 1e-15)
    print(f"\n  Variables with nonzero I_i: {nonzero_vars} / {n}")

    # ------------------------------------------------------------------
    # Part A: Run BINARY-SOBOL
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Part A: BINARY-SOBOL")
    print("=" * 70)

    rng_search = np.random.default_rng(seed + 1)
    t0 = time.time()
    top_k_result, stats = binary_sobol(
        f, n, k, samples_per_estimate=samples_per_estimate,
        rng=rng_search, verbose=False
    )
    elapsed = time.time() - t0

    found_vars = {v for v, _ in top_k_result}
    precision = len(found_vars & true_top_10_vars) / k

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Total evaluations: {stats.total_evaluations:,}")
    print(f"  Groups estimated: {stats.groups_estimated}")
    print(f"  Groups pruned: {stats.groups_pruned}")
    print(f"  Singletons found: {stats.singletons_found}")
    print(f"  Precision@{k}: {precision:.1%}")

    naive_evals = 2 * n * samples_per_estimate
    print(f"  Naive would need: {naive_evals:,} evaluations")
    print(f"  Speedup: {naive_evals / stats.total_evaluations:.1f}x")

    print(f"\n  {'Rank':>4}  {'Var':>5}  {'Est I_i':>10}  {'True I_i':>10}  "
          f"{'Rel Err':>8}  {'In True Top-10?':>15}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*15}")
    for rank, (var, est) in enumerate(top_k_result, 1):
        true_val = true_indices[var]
        rel_err = abs(est - true_val) / true_val if true_val > 0 else float('inf')
        in_true = "YES" if var in true_top_10_vars else "no"
        print(f"  {rank:4d}  {var:5d}  {est:10.6f}  {true_val:10.6f}  "
              f"{rel_err:7.1%}  {in_true:>15}")

    # ------------------------------------------------------------------
    # Part B: Verify estimates via targeted pick-freeze on found variables
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Part B: Verification via targeted pick-freeze on found + true top-10")
    print("=" * 70)

    verify_vars = sorted(found_vars | true_top_10_vars)
    rng_verify = np.random.default_rng(seed + 2)
    verify_samples = 10000

    print(f"\n  Verifying {len(verify_vars)} variables with {verify_samples} "
          f"samples each...")
    print(f"\n  {'Var':>5}  {'True I_i':>10}  {'Binary Est':>10}  "
          f"{'Verify Est':>10}  {'Verify SE':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for var in verify_vars:
        true_val = true_indices[var]

        # Find binary-sobol estimate if available
        bin_est = None
        for v, e in top_k_result:
            if v == var:
                bin_est = e
                break

        ver_est, ver_se, _ = pick_freeze_estimate(
            f, n, [var], verify_samples, rng_verify
        )

        bin_str = f"{bin_est:10.6f}" if bin_est is not None else f"{'--':>10}"
        print(f"  {var:5d}  {true_val:10.6f}  {bin_str}  "
              f"{ver_est:10.6f}  {ver_se:10.6f}")

    # ------------------------------------------------------------------
    # Part C: Fourier coefficient analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Part C: Top Fourier Coefficients (by |c_T|^2)")
    print("=" * 70)

    print(f"\n  {'Rank':>4}  {'|c_T|^2':>10}  {'Degree':>6}  {'Subset':>30}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*30}")
    for rank, (indices, coeff) in enumerate(sorted_coeffs[:20], 1):
        subset_str = "{" + ", ".join(str(int(i)) for i in indices) + "}"
        if len(subset_str) > 30:
            subset_str = subset_str[:27] + "..."
        print(f"  {rank:4d}  {coeff**2:10.6f}  {len(indices):6d}  {subset_str:>30}")

    print(f"\n{'=' * 70}")
    print("Experiment complete.")
    print("=" * 70)


if __name__ == '__main__':
    run_noisy_experiment()
