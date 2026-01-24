"""Tests for sklearn-style TransitionEstimator.

Tests the TransitionEstimator class which provides a scikit-learn style
interface for estimating first-stage transition probabilities in DDC models.
"""

import pytest
import numpy as np
import torch

from econirl.transitions import TransitionEstimator
from econirl.core.types import Panel, Trajectory


class TestTransitionEstimatorSklearn:
    """Tests for sklearn-style TransitionEstimator."""

    @pytest.fixture
    def simple_panel(self):
        """Create a simple panel for testing.

        Creates trajectories where:
        - Action 0 (keep): state increases by 0, 1, or 2
        - Action 1 (replace): state resets (not used for transitions)
        """
        # Create deterministic transitions for testing
        # Trajectory 1: states increase by 1 each period (action=0)
        traj1 = Trajectory(
            states=torch.tensor([0, 1, 2, 3, 4]),
            actions=torch.tensor([0, 0, 0, 0, 0]),  # all keep
            next_states=torch.tensor([1, 2, 3, 4, 5]),
            individual_id=0,
        )

        # Trajectory 2: mix of increments
        traj2 = Trajectory(
            states=torch.tensor([0, 1, 3, 4, 6]),
            actions=torch.tensor([0, 0, 0, 0, 0]),  # all keep
            next_states=torch.tensor([1, 3, 4, 6, 7]),  # +1, +2, +1, +2, +1
            individual_id=1,
        )

        # Trajectory 3: includes a replacement (action=1)
        traj3 = Trajectory(
            states=torch.tensor([5, 6, 0, 1, 2]),
            actions=torch.tensor([0, 1, 0, 0, 0]),  # keep, replace, keep, keep, keep
            next_states=torch.tensor([6, 0, 1, 2, 3]),  # +1, reset, +1, +1, +1
            individual_id=2,
        )

        return Panel(trajectories=[traj1, traj2, traj3])

    def test_transition_estimator_fit_returns_self(self, simple_panel):
        """fit() should return self for method chaining."""
        estimator = TransitionEstimator(n_states=90, max_increase=2)
        result = estimator.fit(simple_panel)

        assert result is estimator

    def test_transition_estimator_matrix_(self, simple_panel):
        """After fit(), matrix_ should be a valid transition matrix."""
        estimator = TransitionEstimator(n_states=10, max_increase=2)
        estimator.fit(simple_panel)

        # Check shape
        assert hasattr(estimator, 'matrix_')
        assert estimator.matrix_.shape == (10, 10)

        # Check rows sum to 1
        row_sums = estimator.matrix_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=1e-6)

        # Check non-negative
        assert (estimator.matrix_ >= 0).all()

        # Check structure: for state s, only states s, s+1, s+2 should have positive prob
        # (except for absorbing state at end)
        for s in range(8):  # Check first 8 states
            for s_prime in range(10):
                if s_prime < s or s_prime > s + 2:
                    assert estimator.matrix_[s, s_prime] == 0.0, \
                        f"Unexpected transition from {s} to {s_prime}"

    def test_transition_estimator_probs_(self, simple_panel):
        """After fit(), probs_ should contain theta_0, theta_1, theta_2."""
        estimator = TransitionEstimator(n_states=10, max_increase=2)
        estimator.fit(simple_panel)

        # Check probs_ exists and has correct shape
        assert hasattr(estimator, 'probs_')
        assert len(estimator.probs_) == 3  # theta_0, theta_1, theta_2

        # Check probabilities sum to 1
        np.testing.assert_allclose(sum(estimator.probs_), 1.0, atol=1e-6)

        # Check all non-negative
        assert all(p >= 0 for p in estimator.probs_)

        # In our simple panel:
        # Transitions (excluding replacements):
        # Traj1: 0->1(+1), 1->2(+1), 2->3(+1), 3->4(+1), 4->5(+1) = 5x (+1)
        # Traj2: 0->1(+1), 1->3(+2), 3->4(+1), 4->6(+2), 6->7(+1) = 3x (+1), 2x (+2)
        # Traj3: 5->6(+1), [skip replacement], 0->1(+1), 1->2(+1), 2->3(+1) = 4x (+1)
        # Total: 12x (+1), 2x (+2), 0x (+0)
        # Expected: theta_0=0, theta_1=12/14, theta_2=2/14
        expected_theta_0 = 0.0
        expected_theta_1 = 12.0 / 14.0
        expected_theta_2 = 2.0 / 14.0

        np.testing.assert_allclose(estimator.probs_[0], expected_theta_0, atol=0.01)
        np.testing.assert_allclose(estimator.probs_[1], expected_theta_1, atol=0.01)
        np.testing.assert_allclose(estimator.probs_[2], expected_theta_2, atol=0.01)

    def test_transition_estimator_summary(self, simple_panel):
        """summary() should return a formatted string with transition info."""
        estimator = TransitionEstimator(n_states=10, max_increase=2)
        estimator.fit(simple_panel)

        summary = estimator.summary()

        # Check it's a string
        assert isinstance(summary, str)

        # Check it contains key information
        assert 'theta' in summary.lower() or 'transition' in summary.lower()
        assert str(estimator.n_transitions_) in summary or 'transitions' in summary.lower()

    def test_transition_estimator_n_transitions_(self, simple_panel):
        """n_transitions_ should count the number of valid transitions."""
        estimator = TransitionEstimator(n_states=10, max_increase=2)
        estimator.fit(simple_panel)

        assert hasattr(estimator, 'n_transitions_')
        # 5 + 5 + 4 = 14 transitions (excluding the replacement in traj3)
        assert estimator.n_transitions_ == 14

    def test_transition_estimator_build_matrix(self):
        """_build_matrix should construct correct transition matrix from probs."""
        estimator = TransitionEstimator(n_states=5, max_increase=2)

        probs = (0.3, 0.5, 0.2)  # theta_0, theta_1, theta_2
        matrix = estimator._build_matrix(probs)

        # Check shape
        assert matrix.shape == (5, 5)

        # Check rows sum to 1
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-6)

        # Check specific entries for non-absorbing states
        # State 0: can go to 0, 1, 2
        np.testing.assert_allclose(matrix[0, 0], 0.3)
        np.testing.assert_allclose(matrix[0, 1], 0.5)
        np.testing.assert_allclose(matrix[0, 2], 0.2)

        # State 1: can go to 1, 2, 3
        np.testing.assert_allclose(matrix[1, 1], 0.3)
        np.testing.assert_allclose(matrix[1, 2], 0.5)
        np.testing.assert_allclose(matrix[1, 3], 0.2)

        # State 2: can go to 2, 3, 4
        np.testing.assert_allclose(matrix[2, 2], 0.3)
        np.testing.assert_allclose(matrix[2, 3], 0.5)
        np.testing.assert_allclose(matrix[2, 4], 0.2)

    def test_transition_estimator_absorbing_state(self):
        """Last state(s) should be absorbing (stay with probability 1)."""
        estimator = TransitionEstimator(n_states=5, max_increase=2)

        probs = (0.3, 0.5, 0.2)
        matrix = estimator._build_matrix(probs)

        # State 4 (last state) should be absorbing
        np.testing.assert_allclose(matrix[4, 4], 1.0)
        np.testing.assert_allclose(matrix[4, :4].sum(), 0.0)

        # State 3 should also accumulate probability at boundary
        # It can only go to states 3 or 4, so theta_2 probability goes to state 4
        np.testing.assert_allclose(matrix[3, 3], 0.3)
        np.testing.assert_allclose(matrix[3, 4], 0.7)  # 0.5 + 0.2

    def test_fit_without_action_column(self):
        """fit() should work when action column is not needed (all keeps)."""
        # All transitions are keeps
        traj = Trajectory(
            states=torch.tensor([0, 1, 2]),
            actions=torch.tensor([0, 0, 0]),
            next_states=torch.tensor([1, 2, 3]),
            individual_id=0,
        )
        panel = Panel(trajectories=[traj])

        estimator = TransitionEstimator(n_states=10, max_increase=2)
        estimator.fit(panel)

        assert hasattr(estimator, 'probs_')
        assert hasattr(estimator, 'matrix_')

    def test_default_parameters(self):
        """Default parameters should match Rust (1987) specification."""
        estimator = TransitionEstimator()

        assert estimator.n_states == 90
        assert estimator.max_increase == 2

    def test_fit_clamps_large_increments(self):
        """Increments larger than max_increase should be clamped."""
        # Trajectory with a large jump
        traj = Trajectory(
            states=torch.tensor([0, 5]),  # Jump of 5
            actions=torch.tensor([0, 0]),
            next_states=torch.tensor([5, 8]),  # Another jump
            individual_id=0,
        )
        panel = Panel(trajectories=[traj])

        estimator = TransitionEstimator(n_states=20, max_increase=2)
        estimator.fit(panel)

        # Large jumps should be clamped to max_increase (2)
        assert estimator.probs_[2] > 0  # theta_2 should have probability
        np.testing.assert_allclose(sum(estimator.probs_), 1.0, atol=1e-6)
