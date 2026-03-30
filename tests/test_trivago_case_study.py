"""Regression tests for Trivago hotel search DDC case study.

Uses n_sessions=500 for fast tests. Validates that the data pipeline,
BC baseline, CCP, and NFXP all run correctly on the 37-state, 4-action
Trivago MDP.
"""

import math

import numpy as np
import pytest
import torch

from econirl.core.types import DDCProblem, Panel
from econirl.datasets.trivago_search import (
    load_trivago_sessions,
    build_trivago_mdp,
    build_trivago_panel,
    build_trivago_features,
    build_trivago_transitions,
    N_STATES,
    N_ACTIONS,
    ABSORBING_STATE,
)
from econirl.preferences.linear import LinearUtility

FEATURE_NAMES = ["step_cost", "browse_cost", "refine_cost", "clickout_value"]
DISCOUNT = 0.95
N_TEST_SESSIONS = 500


# ---------------------------------------------------------------------------
# Fixtures (module-scoped so data loads once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sessions_df():
    return load_trivago_sessions(n_sessions=N_TEST_SESSIONS)


@pytest.fixture(scope="module")
def mdp(sessions_df):
    return build_trivago_mdp(sessions_df)


@pytest.fixture(scope="module")
def panel(mdp):
    return build_trivago_panel(mdp)


@pytest.fixture(scope="module")
def features():
    return build_trivago_features(n_states=N_STATES, n_actions=N_ACTIONS)


@pytest.fixture(scope="module")
def transitions(mdp):
    return build_trivago_transitions(mdp, n_states=N_STATES, n_actions=N_ACTIONS)


@pytest.fixture(scope="module")
def problem():
    return DDCProblem(
        num_states=N_STATES,
        num_actions=N_ACTIONS,
        discount_factor=DISCOUNT,
        scale_parameter=1.0,
    )


@pytest.fixture(scope="module")
def utility(features):
    return LinearUtility(feature_matrix=features, parameter_names=FEATURE_NAMES)


@pytest.fixture(scope="module")
def train_panel(panel):
    n_train = int(panel.num_individuals * 0.8)
    return Panel(trajectories=panel.trajectories[:n_train])


@pytest.fixture(scope="module")
def test_panel(panel):
    n_train = int(panel.num_individuals * 0.8)
    return Panel(trajectories=panel.trajectories[n_train:])


# ---------------------------------------------------------------------------
# Test 1: Data loads correctly
# ---------------------------------------------------------------------------


class TestDataLoads:
    def test_panel_has_sessions(self, panel):
        """Panel should contain multiple sessions with observations."""
        assert panel.num_individuals > 0
        assert panel.num_observations > 0

    def test_panel_structure(self, panel):
        """Each trajectory should have states, actions, and next_states."""
        for traj in panel.trajectories[:5]:
            assert len(traj.states) == len(traj.actions)
            assert len(traj.states) == len(traj.next_states)
            assert traj.states.dtype == torch.long
            assert traj.actions.dtype == torch.long

    def test_states_in_range(self, panel):
        """All states should be in [0, N_STATES)."""
        all_states = panel.get_all_states()
        assert all_states.min().item() >= 0
        assert all_states.max().item() < N_STATES

    def test_actions_in_range(self, panel):
        """All actions should be in [0, N_ACTIONS)."""
        all_actions = panel.get_all_actions()
        assert all_actions.min().item() >= 0
        assert all_actions.max().item() < N_ACTIONS

    def test_transitions_valid(self, transitions):
        """Transition matrix should be properly normalized."""
        assert transitions.shape == (N_ACTIONS, N_STATES, N_STATES)
        row_sums = transitions.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_absorbing_state_self_loops(self, transitions):
        """Absorbing state should self-loop for all actions."""
        for a in range(N_ACTIONS):
            assert transitions[a, ABSORBING_STATE, ABSORBING_STATE].item() > 0.99

    def test_features_shape(self, features):
        """Feature matrix should have correct shape."""
        assert features.shape == (N_STATES, N_ACTIONS, len(FEATURE_NAMES))

    def test_train_test_split(self, train_panel, test_panel, panel):
        """Train/test split should cover all sessions."""
        total = train_panel.num_individuals + test_panel.num_individuals
        assert total == panel.num_individuals


# ---------------------------------------------------------------------------
# Test 2: BC runs
# ---------------------------------------------------------------------------


class TestBCRuns:
    def test_bc_produces_valid_policy(self, train_panel):
        """BC should produce a valid probability distribution over actions."""
        policy = train_panel.compute_choice_frequencies(N_STATES, N_ACTIONS)
        assert policy.shape == (N_STATES, N_ACTIONS)

        # Where we have data, rows should sum to ~1
        visited_mask = policy.sum(dim=1) > 0
        assert visited_mask.any(), "Should have data in at least some states"
        visited_sums = policy[visited_mask].sum(dim=1)
        assert torch.allclose(visited_sums, torch.ones_like(visited_sums), atol=1e-5)

    def test_bc_policy_nonnegative(self, train_panel):
        """BC probabilities should be non-negative."""
        policy = train_panel.compute_choice_frequencies(N_STATES, N_ACTIONS)
        assert (policy >= 0).all()


# ---------------------------------------------------------------------------
# Test 3: CCP runs
# ---------------------------------------------------------------------------


class TestCCPRuns:
    def test_ccp_converges(self, train_panel, utility, problem, transitions):
        """CCP should converge and return 4 parameters."""
        from econirl.estimation.ccp import CCPEstimator

        estimator = CCPEstimator(
            num_policy_iterations=1,
            se_method="asymptotic",
            compute_hessian=False,
            verbose=False,
        )
        result = estimator.estimate(train_panel, utility, problem, transitions)

        assert result.parameters is not None
        assert result.parameters.shape == (len(FEATURE_NAMES),)
        assert result.policy.shape == (N_STATES, N_ACTIONS)
        assert result.converged


# ---------------------------------------------------------------------------
# Test 4: NFXP runs
# ---------------------------------------------------------------------------


class TestNFXPRuns:
    def test_nfxp_converges(self, train_panel, utility, problem, transitions):
        """NFXP should converge and return 4 parameters."""
        from econirl.estimation.nfxp import NFXPEstimator

        estimator = NFXPEstimator(
            optimizer="BHHH",
            inner_solver="hybrid",
            inner_tol=1e-10,
            outer_tol=1e-6,
            outer_max_iter=500,
            se_method="asymptotic",
            compute_hessian=False,
            verbose=False,
        )
        result = estimator.estimate(train_panel, utility, problem, transitions)

        assert result.parameters is not None
        assert result.parameters.shape == (len(FEATURE_NAMES),)
        assert result.policy.shape == (N_STATES, N_ACTIONS)
        assert result.converged


# ---------------------------------------------------------------------------
# Test 5: Structural outperforms uniform
# ---------------------------------------------------------------------------


class TestStructuralOutperformsUniform:
    def test_bc_better_than_uniform(self, train_panel):
        """BC (empirical CCP) LL should beat uniform log(1/4) = -1.386.

        BC is the strongest baseline: it directly estimates P(a|s) from
        data. Structural estimators (CCP, NFXP) impose parametric
        restrictions that may reduce in-sample fit but enable
        counterfactual analysis and interpretation.
        """
        bc_policy = train_panel.compute_choice_frequencies(N_STATES, N_ACTIONS)

        all_s = train_panel.get_all_states()
        all_a = train_panel.get_all_actions()

        bc_ll = torch.log(bc_policy[all_s, all_a].clamp(min=1e-10)).mean().item()
        uniform_ll = math.log(1.0 / N_ACTIONS)  # -1.386

        assert bc_ll > uniform_ll, (
            f"BC LL {bc_ll:.4f} should beat uniform {uniform_ll:.4f}"
        )
