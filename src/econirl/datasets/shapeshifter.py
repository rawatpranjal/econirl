"""Loader for the shape-shifting synthetic DGP.

This is a thin wrapper that constructs a ``ShapeshifterEnvironment``
from the matrix-cell config and simulates a panel for estimator
benchmarking. The environment is the source of truth for ground
truth; this module just exposes the standard loader interface so the
JSS deep-run worker can invoke it the same way it invokes the real
datasets.

The shape-shifter is *not* a real dataset. It exists to verify that
each estimator implementation matches the formulas in its source
paper across the eight axes documented in the environment module.
"""

from __future__ import annotations

from typing import Any

from econirl.core.types import Panel
from econirl.environments.shapeshifter import (
    ShapeshifterConfig,
    ShapeshifterEnvironment,
)


def load_shapeshifter(
    seed: int = 42,
    n_individuals: int = 500,
    n_periods: int = 100,
    as_panel: bool = True,
    **config_kwargs: Any,
) -> Panel:
    """Construct a shape-shifter environment and simulate a panel.

    Parameters
    ----------
    seed : int
        Random seed. Sets both the environment's frozen-network and
        transition-matrix seed (via ``ShapeshifterConfig.seed``) and
        the simulation's trajectory sampler seed.
    n_individuals : int
        Number of trajectories to simulate.
    n_periods : int
        Number of time periods per trajectory. Ignored when the
        environment is finite-horizon (the env terminates earlier).
    as_panel : bool
        Always treated as ``True``; the worker pipeline expects a
        Panel. Argument retained for signature compatibility with
        the other loaders in this package.
    **config_kwargs
        Forwarded to ``ShapeshifterConfig``. Any axis flag (for
        example ``reward_type="neural"`` or ``state_dim=2``) goes
        here. The cell defines what to pass.

    Returns
    -------
    Panel
        Simulated panel ready to feed into an estimator.
    """
    config = ShapeshifterConfig(seed=seed, **config_kwargs)
    env = ShapeshifterEnvironment(config)
    return env.generate_panel(
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )


def get_shapeshifter_info() -> dict[str, Any]:
    """Return metadata about the shape-shifter loader."""
    return {
        "name": "shapeshifter",
        "kind": "synthetic",
        "purpose": "code-vs-paper alignment benchmark",
        "axes": [
            "reward_type",
            "feature_type",
            "action_dependent",
            "stochastic_transitions",
            "stochastic_rewards",
            "num_periods",
            "discount_factor",
            "state_dim",
        ],
    }
