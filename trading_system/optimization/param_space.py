"""
Parameter space definitions and utilities.

Manages parameter grids, random sampling, and parameter perturbation.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def generate_grid(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    """Generate all combinations from a parameter grid (Cartesian product)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def sample_random(
    param_grid: dict[str, list],
    n_samples: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Randomly sample from parameter grid."""
    rng = np.random.RandomState(seed)
    samples = []
    keys = list(param_grid.keys())

    for _ in range(n_samples):
        sample = {}
        for key in keys:
            sample[key] = rng.choice(param_grid[key])
        samples.append(sample)

    return samples


def perturb_params(
    params: dict[str, Any],
    param_ranges: dict[str, list] | None = None,
    n_steps: int = 5,
    perturbation_pct: float = 0.3,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate perturbed versions of parameters for robustness testing.

    For numeric params, varies by ±perturbation_pct.
    For categorical params, tests all values.
    """
    rng = np.random.RandomState(seed)
    perturbed = [params.copy()]

    for key, value in params.items():
        if isinstance(value, (int, float)):
            # Numeric parameter: perturb around the value
            if isinstance(value, int):
                delta = max(1, int(abs(value) * perturbation_pct))
                new_values = list(range(
                    max(1, value - delta * n_steps // 2),
                    value + delta * n_steps // 2 + 1,
                    max(1, delta),
                ))
                new_values = [v for v in new_values if v != value]
            else:
                delta = abs(value) * perturbation_pct
                new_values = np.linspace(
                    max(0.001, value - delta * n_steps / 2),
                    value + delta * n_steps / 2,
                    n_steps,
                )
                new_values = [v for v in new_values if abs(v - value) > delta * 0.01]

            for nv in new_values[:n_steps]:
                p = params.copy()
                p[key] = type(value)(nv)
                perturbed.append(p)

    return perturbed


def count_combinations(param_grid: dict[str, list]) -> int:
    """Count total number of parameter combinations."""
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count


def get_param_subset(
    param_grid: dict[str, list],
    keys: list[str],
) -> dict[str, list]:
    """Get a subset of the parameter grid."""
    return {k: v for k, v in param_grid.items() if k in keys}
