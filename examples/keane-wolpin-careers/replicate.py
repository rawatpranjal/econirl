#!/usr/bin/env python3
"""
Keane & Wolpin (1997) Career Decisions — Finite-Horizon NFXP Replication
========================================================================

Estimates a simplified version of the KW (1997) career choice model using
finite-horizon NFXP with backward induction.

Model:
    - 4 choices: school(0), white-collar(1), blue-collar(2), home(3)
    - State: (schooling, exp_wc, exp_bc) — 704 discrete states
    - Deterministic transitions: school→schooling+1, WC→exp_wc+1, etc.
    - 10 periods (ages 17-26)
    - Logit shocks (Type I EV) — simplification from paper's MV normal
    - Linear utility in state features

Simplification: Paper uses multivariate normal shocks (requires GHK
simulator). We use logit shocks with our existing softmax framework.
Estimates won't match paper exactly but the finite-horizon DDC
structure and NFXP algorithm are identical.

Usage:
    python examples/keane-wolpin-careers/replicate.py
"""

import time

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import backward_induction
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.datasets import load_keane_wolpin
from econirl.estimation.nfxp import NFXPEstimator


# === State encoding ===

# Ranges from data
SCHOOL_MIN, SCHOOL_MAX = 10, 20  # 11 levels
EXP_WC_MAX = 7  # 8 levels (0-7)
EXP_BC_MAX = 7  # 8 levels (0-7)
N_SCHOOL = SCHOOL_MAX - SCHOOL_MIN + 1  # 11
N_EXP_WC = EXP_WC_MAX + 1  # 8
N_EXP_BC = EXP_BC_MAX + 1  # 8
N_STATES = N_SCHOOL * N_EXP_WC * N_EXP_BC  # 704
N_ACTIONS = 4
N_PERIODS = 10
BETA = 0.95


def encode_state(schooling: int, exp_wc: int, exp_bc: int) -> int:
    """Encode (schooling, exp_wc, exp_bc) as a single state index."""
    s_idx = schooling - SCHOOL_MIN
    return s_idx * N_EXP_WC * N_EXP_BC + exp_wc * N_EXP_BC + exp_bc


def decode_state(state: int) -> tuple[int, int, int]:
    """Decode state index back to (schooling, exp_wc, exp_bc)."""
    s_idx = state // (N_EXP_WC * N_EXP_BC)
    rem = state % (N_EXP_WC * N_EXP_BC)
    exp_wc = rem // N_EXP_BC
    exp_bc = rem % N_EXP_BC
    return s_idx + SCHOOL_MIN, exp_wc, exp_bc


def build_transition_matrix() -> jnp.ndarray:
    """Build deterministic transition matrices for KW model.

    Choice 0 (school): schooling += 1 (capped at SCHOOL_MAX)
    Choice 1 (WC work): exp_wc += 1 (capped at EXP_WC_MAX)
    Choice 2 (BC work): exp_bc += 1 (capped at EXP_BC_MAX)
    Choice 3 (home): no state change
    """
    trans = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=np.float64)

    for s in range(N_STATES):
        school, exp_wc, exp_bc = decode_state(s)

        # Choice 0: school -> schooling += 1
        new_school = min(school + 1, SCHOOL_MAX)
        trans[0, s, encode_state(new_school, exp_wc, exp_bc)] = 1.0

        # Choice 1: WC work -> exp_wc += 1
        new_wc = min(exp_wc + 1, EXP_WC_MAX)
        trans[1, s, encode_state(school, new_wc, exp_bc)] = 1.0

        # Choice 2: BC work -> exp_bc += 1
        new_bc = min(exp_bc + 1, EXP_BC_MAX)
        trans[2, s, encode_state(school, exp_wc, new_bc)] = 1.0

        # Choice 3: home -> no change
        trans[3, s, s] = 1.0

    return jnp.array(trans)


def build_feature_matrix() -> jnp.ndarray:
    """Build feature matrix for linear utility specification.

    Features capture returns to schooling, experience, and fixed costs:
    - school_return: schooling level (for school action)
    - wc_exp_return: WC experience (for WC action)
    - bc_exp_return: BC experience (for BC action)
    - school_cost: fixed cost of attending school
    - wc_intercept: base WC wage
    - bc_intercept: base BC wage
    - home_value: value of home production
    """
    n_features = 7
    features = np.zeros((N_STATES, N_ACTIONS, n_features))

    for s in range(N_STATES):
        school, exp_wc, exp_bc = decode_state(s)

        # School returns (action 0)
        features[s, 0, 0] = school / SCHOOL_MAX  # normalized schooling
        features[s, 0, 3] = -1.0  # school cost

        # WC work returns (action 1)
        features[s, 1, 0] = school / SCHOOL_MAX  # schooling affects wages
        features[s, 1, 1] = exp_wc / EXP_WC_MAX  # WC experience
        features[s, 1, 4] = 1.0  # WC intercept

        # BC work returns (action 2)
        features[s, 2, 0] = school / SCHOOL_MAX * 0.5  # less schooling effect
        features[s, 2, 2] = exp_bc / EXP_BC_MAX  # BC experience
        features[s, 2, 5] = 1.0  # BC intercept

        # Home production (action 3)
        features[s, 3, 6] = 1.0  # home value

    return jnp.array(features)


def load_data_as_panel() -> Panel:
    """Load KW data and convert to Panel with compact state encoding."""
    df = load_keane_wolpin()

    trajectories = []
    for ind_id in sorted(df["id"].unique()):
        ind_data = df[df["id"] == ind_id].sort_values("period")

        states = jnp.array(
            [encode_state(int(r["schooling"]), int(r["exp_white_collar"]), int(r["exp_blue_collar"]))
             for _, r in ind_data.iterrows()],
            dtype=jnp.int32,
        )
        actions = jnp.array(ind_data["choice"].values, dtype=jnp.int32)
        next_states = jnp.concatenate([states[1:], states[-1:]])

        trajectories.append(Trajectory(
            states=states, actions=actions, next_states=next_states,
            individual_id=int(ind_id),
        ))

    return Panel(trajectories=trajectories)


class KWUtility:
    """Linear utility specification for KW model."""

    def __init__(self, feature_matrix: jnp.ndarray):
        self._features = feature_matrix
        self.num_parameters = feature_matrix.shape[2]
        self.parameter_names = [
            "school_return", "wc_exp_return", "bc_exp_return",
            "school_cost", "wc_intercept", "bc_intercept", "home_value",
        ]

    @property
    def feature_matrix(self):
        return self._features

    def compute(self, params: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("sak,k->sa", self._features, params)

    def get_initial_parameters(self) -> jnp.ndarray:
        return jnp.zeros(self.num_parameters)

    def get_parameter_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        lower = jnp.full((self.num_parameters,), -20.0)
        upper = jnp.full((self.num_parameters,), 20.0)
        return lower, upper


def main():
    print("=" * 72)
    print("Keane & Wolpin (1997) Career Decisions — NFXP Replication")
    print("=" * 72)
    print("Finite-horizon backward induction, logit shocks, 704 states")
    print()

    # Load data
    df = load_keane_wolpin()
    panel = load_data_as_panel()
    print(f"Data: {len(df):,} obs, {df['id'].nunique()} individuals, {N_PERIODS} periods")
    print(f"State space: {N_STATES} states ({N_SCHOOL} school x {N_EXP_WC} exp_wc x {N_EXP_BC} exp_bc)")
    print(f"Choices: school(0)={sum(df['choice']==0)}, WC(1)={sum(df['choice']==1)}, "
          f"BC(2)={sum(df['choice']==2)}, home(3)={sum(df['choice']==3)}")
    print()

    # Build model components
    transitions = build_transition_matrix()
    features = build_feature_matrix()
    utility = KWUtility(features)
    problem = DDCProblem(
        num_states=N_STATES,
        num_actions=N_ACTIONS,
        discount_factor=BETA,
        num_periods=N_PERIODS,
    )

    # Estimate
    print("Running finite-horizon NFXP...")
    t0 = time.time()
    nfxp = NFXPEstimator(
        optimizer="L-BFGS-B",
        inner_solver="policy",
        compute_hessian=True,
        verbose=True,
        outer_tol=1e-6,
        outer_max_iter=500,
    )
    result = nfxp.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0

    print(f"\nTime: {elapsed:.1f}s")
    print(result.summary())

    # Interpret parameters
    print("\nParameter Interpretation:")
    print("-" * 50)
    for name, val in zip(utility.parameter_names, np.asarray(result.parameters)):
        print(f"  {name:<20} {val:>8.4f}")

    print(f"\nNote: Uses logit shocks (not MV normal from paper).")
    print(f"Estimates demonstrate finite-horizon NFXP on multi-dimensional DDC.")


if __name__ == "__main__":
    main()
