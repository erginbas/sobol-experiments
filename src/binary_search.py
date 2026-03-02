"""BINARY-SOBOL: Adaptive branch-and-bound algorithm for top-k Sobol indices.

Exploits the monotone submodular structure of I_S to find the k most
influential variables using O(k log(n/k)) group estimations instead of O(n).

Key properties used:
- Monotonicity: {i} subset S => I_{i} <= I_S (upper bound for pruning)
- Subadditivity: max_{i in S} I_{i} >= I_S / |S| (existence guarantee)
"""

import heapq
from dataclasses import dataclass, field

import numpy as np

from .estimators import pick_freeze_estimate


@dataclass
class SearchStats:
    """Diagnostics from a BINARY-SOBOL run."""
    total_evaluations: int = 0
    groups_estimated: int = 0
    groups_pruned: int = 0
    singletons_found: int = 0


@dataclass(order=True)
class _QueueItem:
    """Priority queue entry. Negated estimate for max-heap via min-heap."""
    priority: float
    indices: list = field(compare=False)
    estimate: float = field(compare=False)
    std_error: float = field(compare=False)


def binary_sobol(f_callable, n, k, samples_per_estimate=1024,
                 prune_z=2.0, rng=None, verbose=False):
    """Find the k variables with the largest total Sobol indices.

    Algorithm:
    1. Estimate Var(f) and bootstrap tau by probing random singletons.
    2. Push [n] into a max-priority queue.
    3. Pop the highest-priority group, split, estimate halves.
    4. Prune halves whose upper confidence bound <= tau.
    5. Record singletons; update tau = k-th largest singleton so far.
    6. Repeat until all groups are resolved.

    Parameters
    ----------
    f_callable : callable
        Function taking binary array (m, n) -> array (m,).
    n : int
        Number of variables.
    k : int
        Number of top variables to find.
    samples_per_estimate : int
        Samples per pick-freeze estimation.
    prune_z : float
        Number of standard errors for confidence-based pruning.
        A group is pruned if estimate + prune_z * std_error <= tau.
    rng : np.random.Generator, optional
    verbose : bool

    Returns
    -------
    top_k : list of (variable_index, estimated_I)
        Top-k variables sorted by estimated index, descending.
    stats : SearchStats
    """
    if rng is None:
        rng = np.random.default_rng()

    stats = SearchStats()
    singletons = {}  # variable_index -> (estimate, std_error)

    # Step 1: Estimate Var(f) = I_{[n]}
    all_indices = list(range(n))
    var_est, var_se, ne = pick_freeze_estimate(f_callable, n, all_indices,
                                               samples_per_estimate, rng)
    stats.total_evaluations += ne
    stats.groups_estimated += 1

    if verbose:
        print(f"Var(f) estimate: {var_est:.6f} +/- {var_se:.6f}")

    # Step 2: Bootstrap tau by probing random singletons.
    # Sample min(2k, n) random variables to get an initial threshold.
    num_probes = min(2 * k, n)
    probe_indices = rng.choice(n, size=num_probes, replace=False).tolist()

    for idx in probe_indices:
        est_i, se_i, ne_i = pick_freeze_estimate(
            f_callable, n, [idx], samples_per_estimate, rng
        )
        stats.total_evaluations += ne_i
        stats.groups_estimated += 1
        singletons[idx] = (est_i, se_i)
        stats.singletons_found += 1

    tau = _get_tau(singletons, k)

    if verbose:
        print(f"Bootstrap: probed {num_probes} singletons, tau={tau:.6f}")

    # Step 3: Initialize priority queue with full set
    # But skip variables already probed — they're resolved as singletons.
    remaining = [i for i in range(n) if i not in singletons]

    if not remaining:
        # All variables already probed
        sorted_vars = sorted(singletons.items(), key=lambda x: -x[1][0])
        top_k = [(var, est) for var, (est, se) in sorted_vars[:k]]
        return top_k, stats

    # Estimate I for the remaining set and push to queue
    rem_est, rem_se, rem_ne = pick_freeze_estimate(
        f_callable, n, remaining, samples_per_estimate, rng
    )
    stats.total_evaluations += rem_ne
    stats.groups_estimated += 1

    queue = []
    _maybe_push(queue, remaining, rem_est, rem_se, tau, prune_z, stats, verbose)

    # Step 4: Binary search loop
    while queue:
        item = heapq.heappop(queue)
        indices = item.indices
        est = item.estimate
        se = item.std_error

        # Re-check pruning (tau may have increased since this was enqueued)
        upper_bound = est + prune_z * se
        if upper_bound <= tau:
            stats.groups_pruned += 1
            if verbose:
                print(f"  Pruned group of size {len(indices)}, "
                      f"UB={upper_bound:.6f} <= tau={tau:.6f}")
            continue

        # Singleton: record it
        if len(indices) == 1:
            var = indices[0]
            singletons[var] = (est, se)
            stats.singletons_found += 1
            tau = max(tau, _get_tau(singletons, k))
            if verbose:
                print(f"  Singleton var={var}, I={est:.6f} +/- {se:.6f}, "
                      f"tau={tau:.6f}")
            continue

        # Split into two halves
        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        # Estimate both halves
        for half in [left, right]:
            h_est, h_se, h_ne = pick_freeze_estimate(
                f_callable, n, half, samples_per_estimate, rng
            )
            stats.total_evaluations += h_ne
            stats.groups_estimated += 1

            if len(half) == 1:
                var = half[0]
                h_upper = h_est + prune_z * h_se
                if h_upper <= tau:
                    stats.groups_pruned += 1
                    if verbose:
                        print(f"  Pruned singleton var={var}, "
                              f"UB={h_upper:.6f} <= tau={tau:.6f}")
                    continue
                singletons[var] = (h_est, h_se)
                stats.singletons_found += 1
                tau = max(tau, _get_tau(singletons, k))
                if verbose:
                    print(f"  Singleton var={var}, I={h_est:.6f} +/- {h_se:.6f}")
            else:
                _maybe_push(queue, half, h_est, h_se, tau, prune_z,
                            stats, verbose)

    # Collect results: top-k by estimated index
    sorted_vars = sorted(singletons.items(), key=lambda x: -x[1][0])
    top_k = [(var, est) for var, (est, se) in sorted_vars[:k]]

    return top_k, stats


def _maybe_push(queue, indices, est, se, tau, prune_z, stats, verbose):
    """Push a group onto the queue, or prune it."""
    upper_bound = est + prune_z * se
    if upper_bound <= tau:
        stats.groups_pruned += 1
        if verbose:
            print(f"  Pruned group of size {len(indices)}, "
                  f"UB={upper_bound:.6f} <= tau={tau:.6f}")
        return
    heapq.heappush(queue, _QueueItem(-est, indices, est, se))


def _get_tau(singletons, k):
    """Get the k-th largest singleton estimate (pruning threshold)."""
    if len(singletons) < k:
        return 0.0
    estimates = [est for est, se in singletons.values()]
    estimates.sort(reverse=True)
    return estimates[k - 1]
