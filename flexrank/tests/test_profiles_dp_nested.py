"""Tests for exact nested filtering in profiles.dp."""

from itertools import product

import numpy as np

from flexrank.profiles.dp import minimize_error_with_any_drops as dp_profile


def normalize_solutions(solutions):
    """Convert drop vectors to tuples and round floats for stable comparison."""
    return sorted(
        [(round(error, 8), tuple(drops)) for error, drops in solutions],
        key=lambda item: (item[0], item[1]),
    )


def brute_frontier(savings_errors):
    """Enumerate exact Pareto solutions for a tiny DP instance."""
    choices = [list(zip(savings, errors)) + [(0, 0.0)] for savings, errors in savings_errors]
    solutions = []

    for combo in product(*choices):
        drops = [saving for saving, _ in combo]
        error = sum(err for _, err in combo)
        total_savings = sum(drops)
        solutions.append((total_savings, error, drops))

    best_per_total = {}
    for total_savings, error, drops in solutions:
        if total_savings not in best_per_total or error < best_per_total[total_savings][0]:
            best_per_total[total_savings] = (error, drops)

    ordered = sorted(
        (total_savings, error, drops) for total_savings, (error, drops) in best_per_total.items()
    )
    filtered = []
    best_err_from_right = float("inf")
    for total_savings, error, drops in reversed(ordered):
        if error < best_err_from_right:
            filtered.append((error, drops))
            best_err_from_right = error

    return list(reversed(filtered))


def brute_nested_filter(solutions):
    """Apply the exact nested filter to final Pareto solutions."""
    kept = []
    last_kept = None

    for error, drops in solutions:
        cur_drops = np.array(drops, dtype=int)
        if last_kept is None or np.all(cur_drops >= last_kept):
            kept.append((error, drops))
            last_kept = cur_drops

    return kept


def test_profiles_dp_nested_keeps_only_nested_solutions():
    """Nested mode should drop final solutions that break the chain."""
    savings_errors = [
        ([4], [0.2]),
        ([3], [0.1]),
    ]

    default = dp_profile(savings_errors, nested=False)
    nested = dp_profile(savings_errors, nested=True)

    assert default == [
        (0.0, [0, 0]),
        (0.1, [0, 3]),
        (0.2, [4, 0]),
        (0.30000000000000004, [4, 3]),
    ]
    assert nested == [
        (0.0, [0, 0]),
        (0.1, [0, 3]),
        (0.30000000000000004, [4, 3]),
    ]

    nested_drops = [np.array(drops) for _, drops in nested]
    for prev, cur in zip(nested_drops, nested_drops[1:]):
        assert np.all(cur >= prev)


def test_profiles_dp_nested_matches_exact_post_filter():
    """Nested mode should equal exact post-filtering of final Pareto solutions."""
    rng = np.random.default_rng(7)

    for _ in range(30):
        savings_errors = []
        for _ in range(4):
            n_opts = int(rng.integers(1, 4))
            savings = rng.integers(1, 8, size=n_opts).tolist()
            errors = (rng.random(n_opts) * 3.0).tolist()
            savings_errors.append((savings, errors))

        exact = brute_nested_filter(brute_frontier(savings_errors))
        got = dp_profile(savings_errors, nested=True)

        assert normalize_solutions(got) == normalize_solutions(exact)
