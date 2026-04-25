"""Source of truth for the JSS deep-run experiment matrix.

Every numerical artifact in the paper traces to a Cell defined here.
The dispatcher reads this file and fans out one job per cell. The
aggregator and reporter consume the per-cell CSVs and the headline
tags declared on each cell.

Cell ids follow the convention `<tier>_<dataset>_<estimator>` so the
dispatcher's per-pod logs sort sensibly. A small number of cells need
to disambiguate further (the GPU-speedup tier reuses the same
estimator-dataset pair on two hardware targets), and those carry an
explicit suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Hardware = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class Cell:
    """One benchmark cell of the matrix.

    Attributes:
        cell_id: Unique identifier. Used as the per-pod log directory
            name and as the per-cell output CSV filename.
        estimator: Estimator name from `econirl.estimation`. The worker
            resolves this to a concrete class via a small registry.
        dataset: Dataset name from `econirl.datasets`. The worker
            resolves this to a loader call via a small registry.
        hardware: Which device to run on. Used by the dispatcher to
            pick the RunPod GPU type and by the worker to set the
            JAX platform.
        n_replications: Number of Monte Carlo replications. Each
            replication is a fresh fit with seed `seed_base + r`.
        headline_tag: Which headline this cell contributes to. Used
            by the aggregator to route the result into the right
            headline CSV.
        expected_runtime_s: Per-replication wall-clock budget in
            seconds. The dispatcher uses this to set the pod timeout
            and to schedule longer cells first.
        extra_kwargs: Estimator-specific keyword arguments that the
            worker passes through to the estimator constructor. For
            example, the AIRL-Het cells set `num_segments=2`, the
            transfer cells set `perturb_transitions=True`, and the
            ziebart-big cells set `reward_type='neural'`.
    """

    cell_id: str
    estimator: str
    dataset: str
    hardware: Hardware
    n_replications: int
    headline_tag: str
    expected_runtime_s: int
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier 1. Hero panel. One cell per dataset showing the intended winner.
# ---------------------------------------------------------------------------

TIER_1_HERO = [
    Cell(
        cell_id="tier1_rust_small_nfxp",
        estimator="NFXP",
        dataset="rust-small",
        hardware="cpu",
        n_replications=1,
        headline_tag="hero",
        expected_runtime_s=10,
    ),
    Cell(
        cell_id="tier1_rust_big_gladius",
        estimator="GLADIUS",
        dataset="rust-big",
        hardware="gpu",
        n_replications=1,
        headline_tag="hero",
        expected_runtime_s=120,
    ),
    Cell(
        cell_id="tier1_ziebart_small_mceirl",
        estimator="MCE-IRL",
        dataset="ziebart-small",
        hardware="cpu",
        n_replications=1,
        headline_tag="hero",
        expected_runtime_s=30,
    ),
    Cell(
        cell_id="tier1_ziebart_big_deepmce",
        estimator="MCE-IRL",
        dataset="ziebart-big",
        hardware="gpu",
        n_replications=1,
        headline_tag="hero",
        expected_runtime_s=300,
        extra_kwargs={"reward_type": "neural"},
    ),
    Cell(
        cell_id="tier1_lsw_synthetic_airlhet",
        estimator="AIRL-Het",
        dataset="lsw-synthetic",
        hardware="gpu",
        n_replications=1,
        headline_tag="hero",
        expected_runtime_s=600,
        extra_kwargs={"num_segments": 2},
    ),
]


# ---------------------------------------------------------------------------
# Tier 2. Equivalence on rust-small. Twelve estimators, R=20, CPU.
# ---------------------------------------------------------------------------

_TIER_2_ESTIMATORS = [
    ("NFXP",     30),
    ("CCP",      10),
    ("MPEC",     20),
    ("MCE-IRL",  30),
    ("NNES",     60),
    ("SEES",     30),
    ("TD-CCP",  120),
    ("GLADIUS",  60),
    ("AIRL",    600),
    ("IQ-Learn",  5),
    ("f-IRL",   120),
    ("BC",        2),
]

TIER_2_EQUIVALENCE = [
    Cell(
        cell_id=f"tier2_rust_small_{name.lower().replace('-', '').replace('_', '')}",
        estimator=name,
        dataset="rust-small",
        hardware="cpu",
        n_replications=20,
        headline_tag="A_equivalence",
        expected_runtime_s=runtime,
    )
    for name, runtime in _TIER_2_ESTIMATORS
]


# ---------------------------------------------------------------------------
# Tier 3a. Failure-and-recovery on rust-big. R=20, GPU.
# ---------------------------------------------------------------------------

TIER_3A_FAILURE_RECOVERY = [
    Cell(
        cell_id="tier3a_rust_big_nfxp",
        estimator="NFXP",
        dataset="rust-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="C_failure_recovery",
        expected_runtime_s=300,
        extra_kwargs={"expect_failure": True},
    ),
    Cell(
        cell_id="tier3a_rust_big_ccp",
        estimator="CCP",
        dataset="rust-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="C_failure_recovery",
        expected_runtime_s=300,
        extra_kwargs={"expect_failure": True},
    ),
    Cell(
        cell_id="tier3a_rust_big_gladius",
        estimator="GLADIUS",
        dataset="rust-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="C_failure_recovery",
        expected_runtime_s=120,
    ),
    Cell(
        cell_id="tier3a_rust_big_nnes",
        estimator="NNES",
        dataset="rust-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="C_failure_recovery",
        expected_runtime_s=180,
    ),
    Cell(
        cell_id="tier3a_rust_big_tdccp",
        estimator="TD-CCP",
        dataset="rust-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="C_failure_recovery",
        expected_runtime_s=300,
    ),
]


# ---------------------------------------------------------------------------
# Tier 3b. IRL scalability on ziebart-big. R=20, GPU.
# ---------------------------------------------------------------------------

TIER_3B_IRL_SCALABILITY = [
    Cell(
        cell_id="tier3b_ziebart_big_deepmce",
        estimator="MCE-IRL",
        dataset="ziebart-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="D_irl_scalability",
        expected_runtime_s=300,
        extra_kwargs={"reward_type": "neural"},
    ),
    Cell(
        cell_id="tier3b_ziebart_big_mceirl",
        estimator="MCE-IRL",
        dataset="ziebart-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="D_irl_scalability",
        expected_runtime_s=1200,
        extra_kwargs={"reward_type": "linear"},
    ),
    Cell(
        cell_id="tier3b_ziebart_big_airl",
        estimator="AIRL",
        dataset="ziebart-big",
        hardware="gpu",
        n_replications=20,
        headline_tag="D_irl_scalability",
        expected_runtime_s=1200,
    ),
]


# ---------------------------------------------------------------------------
# Tier 3c. Unobserved heterogeneity on lsw-synthetic. R=20, GPU.
# ---------------------------------------------------------------------------

TIER_3C_HETEROGENEITY = [
    Cell(
        cell_id="tier3c_lsw_synthetic_airlhet",
        estimator="AIRL-Het",
        dataset="lsw-synthetic",
        hardware="gpu",
        n_replications=20,
        headline_tag="E_heterogeneity",
        expected_runtime_s=600,
        extra_kwargs={"num_segments": 2, "exit_action": 2, "absorbing_state": 0},
    ),
    Cell(
        cell_id="tier3c_lsw_synthetic_airl",
        estimator="AIRL",
        dataset="lsw-synthetic",
        hardware="gpu",
        n_replications=20,
        headline_tag="E_heterogeneity",
        expected_runtime_s=300,
    ),
    Cell(
        cell_id="tier3c_lsw_synthetic_mceirl",
        estimator="MCE-IRL",
        dataset="lsw-synthetic",
        hardware="gpu",
        n_replications=20,
        headline_tag="E_heterogeneity",
        expected_runtime_s=120,
    ),
    Cell(
        cell_id="tier3c_lsw_synthetic_bc",
        estimator="BC",
        dataset="lsw-synthetic",
        hardware="cpu",
        n_replications=20,
        headline_tag="E_heterogeneity",
        expected_runtime_s=10,
    ),
]


# ---------------------------------------------------------------------------
# Tier 3d. Transfer on perturbed rust-small. R=50, CPU.
# ---------------------------------------------------------------------------

TIER_3D_TRANSFER = [
    Cell(
        cell_id="tier3d_rust_perturbed_airl",
        estimator="AIRL",
        dataset="rust-small",
        hardware="cpu",
        n_replications=50,
        headline_tag="F_transfer",
        expected_runtime_s=600,
        extra_kwargs={"perturb_transitions": True, "perturb_seed": 7},
    ),
    Cell(
        cell_id="tier3d_rust_perturbed_mceirl",
        estimator="MCE-IRL",
        dataset="rust-small",
        hardware="cpu",
        n_replications=50,
        headline_tag="F_transfer",
        expected_runtime_s=30,
        extra_kwargs={"perturb_transitions": True, "perturb_seed": 7},
    ),
    Cell(
        cell_id="tier3d_rust_perturbed_iqlearn",
        estimator="IQ-Learn",
        dataset="rust-small",
        hardware="cpu",
        n_replications=50,
        headline_tag="F_transfer",
        expected_runtime_s=10,
        extra_kwargs={"perturb_transitions": True, "perturb_seed": 7},
    ),
    Cell(
        cell_id="tier3d_rust_perturbed_firl",
        estimator="f-IRL",
        dataset="rust-small",
        hardware="cpu",
        n_replications=50,
        headline_tag="F_transfer",
        expected_runtime_s=120,
        extra_kwargs={"perturb_transitions": True, "perturb_seed": 7},
    ),
]


# ---------------------------------------------------------------------------
# Tier 3e. GPU speedup. CPU and GPU pairs for the neural estimators.
# R=3 because we are timing rather than recovering parameters.
# ---------------------------------------------------------------------------

_TIER_3E_PAIRS = [
    ("NNES",      "rust-small",     30,  60),
    ("NNES",      "rust-big",      180, 360),
    ("TD-CCP",    "rust-small",    120, 240),
    ("TD-CCP",    "rust-big",      300, 600),
    ("GLADIUS",   "rust-small",     60, 180),
    ("GLADIUS",   "rust-big",      120, 240),
    ("MCE-IRL",   "ziebart-small", 120, 240),
    ("MCE-IRL",   "ziebart-big",   300, 900),
    ("AIRL-Het",  "lsw-synthetic", 600, 1800),
]

TIER_3E_GPU_SPEEDUP = []
for est, ds, gpu_runtime, cpu_runtime in _TIER_3E_PAIRS:
    est_slug = est.lower().replace("-", "").replace("_", "")
    ds_slug = ds.replace("-", "_")
    extra: dict[str, Any] = {}
    if est == "MCE-IRL" and ds == "ziebart-big":
        extra = {"reward_type": "neural"}
    if est == "AIRL-Het":
        extra = {"num_segments": 2, "exit_action": 2, "absorbing_state": 0}
    TIER_3E_GPU_SPEEDUP.extend(
        [
            Cell(
                cell_id=f"tier3e_{ds_slug}_{est_slug}_gpu",
                estimator=est,
                dataset=ds,
                hardware="gpu",
                n_replications=3,
                headline_tag="G_gpu_speedup",
                expected_runtime_s=gpu_runtime,
                extra_kwargs=extra,
            ),
            Cell(
                cell_id=f"tier3e_{ds_slug}_{est_slug}_cpu",
                estimator=est,
                dataset=ds,
                hardware="cpu",
                n_replications=3,
                headline_tag="G_gpu_speedup",
                expected_runtime_s=cpu_runtime,
                extra_kwargs=extra,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Master matrix and tier index
# ---------------------------------------------------------------------------

MATRIX: list[Cell] = (
    TIER_1_HERO
    + TIER_2_EQUIVALENCE
    + TIER_3A_FAILURE_RECOVERY
    + TIER_3B_IRL_SCALABILITY
    + TIER_3C_HETEROGENEITY
    + TIER_3D_TRANSFER
    + TIER_3E_GPU_SPEEDUP
)

TIER_INDEX: dict[str, list[Cell]] = {
    "1":  TIER_1_HERO,
    "2":  TIER_2_EQUIVALENCE,
    "3a": TIER_3A_FAILURE_RECOVERY,
    "3b": TIER_3B_IRL_SCALABILITY,
    "3c": TIER_3C_HETEROGENEITY,
    "3d": TIER_3D_TRANSFER,
    "3e": TIER_3E_GPU_SPEEDUP,
}


def cells_for_tiers(tiers: list[str]) -> list[Cell]:
    """Return the cells belonging to the requested tiers, in order."""
    selected: list[Cell] = []
    for t in tiers:
        if t not in TIER_INDEX:
            raise ValueError(
                f"Unknown tier {t!r}. Choose from {sorted(TIER_INDEX)}."
            )
        selected.extend(TIER_INDEX[t])
    return selected


def get_cell(cell_id: str) -> Cell:
    """Look up a single cell by id."""
    for cell in MATRIX:
        if cell.cell_id == cell_id:
            return cell
    raise KeyError(f"Unknown cell_id {cell_id!r}")


def matrix_summary() -> dict[str, Any]:
    """Return a quick summary of the matrix size and runtime budget."""
    total_cells = len(MATRIX)
    total_fits = sum(c.n_replications for c in MATRIX)
    total_runtime_s = sum(c.n_replications * c.expected_runtime_s for c in MATRIX)
    by_tier = {
        tier: {
            "cells": len(cells),
            "fits": sum(c.n_replications for c in cells),
            "runtime_s": sum(c.n_replications * c.expected_runtime_s for c in cells),
        }
        for tier, cells in TIER_INDEX.items()
    }
    return {
        "total_cells": total_cells,
        "total_fits": total_fits,
        "total_runtime_s": total_runtime_s,
        "total_runtime_h": total_runtime_s / 3600.0,
        "by_tier": by_tier,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(matrix_summary(), indent=2))
