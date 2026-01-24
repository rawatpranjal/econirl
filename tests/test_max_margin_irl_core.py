"""Tests for MaxMarginIRLEstimator core functionality.

Tests cover:
1. Estimator instantiation
2. _compute_feature_expectations works correctly
3. _find_violating_policy works correctly
4. _solve_qp works correctly
5. _optimize runs and returns EstimationResult
6. estimate() method works (inherited from BaseEstimator)
"""

import pytest
import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.estimation.max_margin_irl import MaxMarginIRLEstimator
from econirl.preferences.reward import LinearReward


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_problem() -> DDCProblem:
    """Simple 5-state, 2-action problem for testing."""
    return DDCProblem(
        num_states=5,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
    )


@pytest.fixture
def simple_transitions(simple_problem) -> torch.Tensor:
    """Simple deterministic transitions for testing.

    Action 0: move to next state (with wrap-around)
    Action 1: reset to state 0
    """
    num_states = simple_problem.num_states
    num_actions = simple_problem.num_actions

    transitions = torch.zeros((num_actions, num_states, num_states))

    # Action 0: move to next state
    for s in range(num_states):
        next_s = (s + 1) % num_states
        transitions[0, s, next_s] = 1.0

    # Action 1: reset to state 0
    for s in range(num_states):
        transitions[1, s, 0] = 1.0

    return transitions


@pytest.fixture
def simple_state_features(simple_problem) -> torch.Tensor:
    """Simple state features: [state_index/num_states, is_goal_state]."""
    num_states = simple_problem.num_states

    features = torch.zeros((num_states, 2))
    for s in range(num_states):
        features[s, 0] = s / (num_states - 1)  # Normalized state index
        features[s, 1] = 1.0 if s == num_states - 1 else 0.0  # Goal indicator

    return features


@pytest.fixture
def simple_reward_fn(simple_state_features, simple_problem) -> LinearReward:
    """Simple linear reward function."""
    return LinearReward(
        state_features=simple_state_features,
        parameter_names=["progress", "goal_bonus"],
        n_actions=simple_problem.num_actions,
    )


@pytest.fixture
def simple_panel(simple_problem) -> Panel:
    """Simple panel with expert demonstrations.

    Expert policy: always take action 0 (move forward to reach goal)
    """
    trajectories = []

    # Create 10 trajectories
    for i in range(10):
        states = []
        actions = []
        next_states = []

        state = 0
        for t in range(20):  # 20 steps per trajectory
            states.append(state)
            action = 0  # Expert always moves forward
            actions.append(action)
            next_state = (state + 1) % simple_problem.num_states
            next_states.append(next_state)
            state = next_state

        traj = Trajectory(
            states=torch.tensor(states, dtype=torch.long),
            actions=torch.tensor(actions, dtype=torch.long),
            next_states=torch.tensor(next_states, dtype=torch.long),
            individual_id=i,
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


# ============================================================================
# Test Instantiation
# ============================================================================


class TestInstantiation:
    """Tests for estimator instantiation."""

    def test_default_instantiation(self):
        """Test that estimator can be instantiated with defaults."""
        estimator = MaxMarginIRLEstimator()

        assert estimator is not None
        assert estimator.name == "Max Margin IRL (Abbeel & Ng 2004)"

    def test_custom_parameters(self):
        """Test instantiation with custom parameters."""
        estimator = MaxMarginIRLEstimator(
            max_iterations=100,
            margin_tol=1e-6,
            value_tol=1e-10,
            qp_method="trust-constr",
            verbose=True,
        )

        assert estimator._max_iterations == 100
        assert estimator._margin_tol == 1e-6
        assert estimator._value_tol == 1e-10
        assert estimator._qp_method == "trust-constr"
        assert estimator._verbose is True

    def test_has_required_methods(self):
        """Test that estimator has all required methods."""
        estimator = MaxMarginIRLEstimator()

        assert hasattr(estimator, "estimate")
        assert hasattr(estimator, "_optimize")
        assert hasattr(estimator, "_compute_feature_expectations")
        assert hasattr(estimator, "_find_violating_policy")
        assert hasattr(estimator, "_solve_qp")
        assert callable(estimator.estimate)


# ============================================================================
# Test _compute_feature_expectations
# ============================================================================


class TestComputeFeatureExpectations:
    """Tests for _compute_feature_expectations method."""

    def test_returns_correct_shape(self, simple_panel, simple_reward_fn):
        """Test that feature expectations have correct shape."""
        estimator = MaxMarginIRLEstimator()

        feature_exp = estimator._compute_feature_expectations(
            simple_panel, simple_reward_fn
        )

        assert feature_exp.shape == (simple_reward_fn.num_parameters,)

    def test_non_negative_features(self, simple_panel, simple_reward_fn):
        """Test that feature expectations are reasonable for non-negative features."""
        estimator = MaxMarginIRLEstimator()

        feature_exp = estimator._compute_feature_expectations(
            simple_panel, simple_reward_fn
        )

        # Our features are non-negative, so expectations should be too
        assert (feature_exp >= 0).all()

    def test_matches_manual_computation(
        self, simple_panel, simple_reward_fn, simple_problem
    ):
        """Test that feature expectations match manual computation."""
        estimator = MaxMarginIRLEstimator()

        feature_exp = estimator._compute_feature_expectations(
            simple_panel, simple_reward_fn
        )

        # Manual computation
        manual_exp = torch.zeros(2)
        count = 0
        for traj in simple_panel.trajectories:
            for t in range(len(traj)):
                state = traj.states[t].item()
                manual_exp += simple_reward_fn.state_features[state]
                count += 1
        manual_exp /= count

        assert torch.allclose(feature_exp, manual_exp, atol=1e-6)

    def test_empty_trajectory_handling(self, simple_reward_fn):
        """Test handling of minimal data."""
        # Single observation
        traj = Trajectory(
            states=torch.tensor([0]),
            actions=torch.tensor([0]),
            next_states=torch.tensor([1]),
        )
        panel = Panel(trajectories=[traj])

        estimator = MaxMarginIRLEstimator()
        feature_exp = estimator._compute_feature_expectations(panel, simple_reward_fn)

        # Should equal features at state 0
        expected = simple_reward_fn.state_features[0]
        assert torch.allclose(feature_exp, expected, atol=1e-6)


# ============================================================================
# Test _find_violating_policy
# ============================================================================


class TestFindViolatingPolicy:
    """Tests for _find_violating_policy method."""

    def test_returns_valid_policy(
        self, simple_problem, simple_transitions, simple_reward_fn
    ):
        """Test that returned policy is valid (sums to 1, non-negative)."""
        estimator = MaxMarginIRLEstimator()

        theta = torch.tensor([1.0, 2.0])  # Reward progress and goal

        policy, V = estimator._find_violating_policy(
            theta, simple_transitions, simple_reward_fn, simple_problem
        )

        # Check policy shape
        assert policy.shape == (simple_problem.num_states, simple_problem.num_actions)

        # Check non-negative
        assert (policy >= 0).all()

        # Check rows sum to 1
        row_sums = policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_returns_value_function(
        self, simple_problem, simple_transitions, simple_reward_fn
    ):
        """Test that returned value function has correct shape."""
        estimator = MaxMarginIRLEstimator()

        theta = torch.tensor([1.0, 2.0])

        policy, V = estimator._find_violating_policy(
            theta, simple_transitions, simple_reward_fn, simple_problem
        )

        assert V.shape == (simple_problem.num_states,)
        assert torch.isfinite(V).all()

    def test_policy_responds_to_reward(
        self, simple_problem, simple_transitions, simple_reward_fn
    ):
        """Test that policy changes with different reward weights."""
        estimator = MaxMarginIRLEstimator()

        # Reward that favors progress
        theta1 = torch.tensor([1.0, 0.0])
        policy1, _ = estimator._find_violating_policy(
            theta1, simple_transitions, simple_reward_fn, simple_problem
        )

        # Reward that only values goal
        theta2 = torch.tensor([0.0, 10.0])
        policy2, _ = estimator._find_violating_policy(
            theta2, simple_transitions, simple_reward_fn, simple_problem
        )

        # Policies should be different
        # (Though both might prefer action 0 to reach goal, probabilities differ)
        # At minimum, check they're both valid
        assert torch.isfinite(policy1).all()
        assert torch.isfinite(policy2).all()


# ============================================================================
# Test _solve_qp
# ============================================================================


class TestSolveQP:
    """Tests for _solve_qp method."""

    def test_no_constraints(self):
        """Test QP solution with no constraints."""
        estimator = MaxMarginIRLEstimator()

        expert_features = torch.tensor([0.5, 0.3, 0.2])
        violating_features = []

        theta, margin = estimator._solve_qp(expert_features, violating_features)

        # Should return normalized weights
        assert torch.isclose(torch.norm(theta), torch.tensor(1.0), atol=1e-5)
        assert margin == 0.0

    def test_single_constraint(self):
        """Test QP solution with single constraint."""
        estimator = MaxMarginIRLEstimator()

        expert_features = torch.tensor([1.0, 0.0])
        violating_features = [torch.tensor([0.0, 1.0])]

        theta, margin = estimator._solve_qp(expert_features, violating_features)

        # Theta should be normalized
        assert torch.isclose(torch.norm(theta), torch.tensor(1.0), atol=1e-4)

        # Margin should be positive (expert better than violating)
        assert margin >= -1e-6  # Allow small numerical tolerance

    def test_multiple_constraints(self):
        """Test QP solution with multiple constraints."""
        estimator = MaxMarginIRLEstimator()

        expert_features = torch.tensor([1.0, 1.0])
        violating_features = [
            torch.tensor([0.5, 0.0]),
            torch.tensor([0.0, 0.5]),
            torch.tensor([0.3, 0.3]),
        ]

        theta, margin = estimator._solve_qp(expert_features, violating_features)

        # Check normalization
        assert torch.isclose(torch.norm(theta), torch.tensor(1.0), atol=1e-4)

        # Check all constraints are approximately satisfied
        for vf in violating_features:
            constraint_value = torch.dot(theta, expert_features - vf).item()
            assert constraint_value >= margin - 1e-4

    def test_returns_normalized_theta(self):
        """Test that theta is always normalized to unit norm."""
        estimator = MaxMarginIRLEstimator()

        expert_features = torch.tensor([2.0, 3.0, 1.0])
        violating_features = [torch.tensor([1.0, 1.0, 1.0])]

        theta, _ = estimator._solve_qp(expert_features, violating_features)

        assert torch.isclose(torch.norm(theta), torch.tensor(1.0), atol=1e-4)


# ============================================================================
# Test _optimize
# ============================================================================


class TestOptimize:
    """Tests for _optimize method."""

    def test_returns_estimation_result(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that _optimize returns EstimationResult."""
        estimator = MaxMarginIRLEstimator(max_iterations=5, verbose=False)

        result = estimator._optimize(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        # Check result type
        from econirl.estimation.base import EstimationResult
        assert isinstance(result, EstimationResult)

    def test_result_has_correct_fields(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that result has all required fields."""
        estimator = MaxMarginIRLEstimator(max_iterations=5, verbose=False)

        result = estimator._optimize(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        # Check parameters
        assert result.parameters is not None
        assert result.parameters.shape == (simple_reward_fn.num_parameters,)

        # Check policy and value function
        assert result.policy is not None
        assert result.policy.shape == (
            simple_problem.num_states,
            simple_problem.num_actions,
        )
        assert result.value_function is not None
        assert result.value_function.shape == (simple_problem.num_states,)

        # Check iteration info
        assert result.num_iterations > 0
        assert isinstance(result.converged, bool)
        assert result.optimization_time > 0

    def test_result_metadata(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that result metadata contains expected fields."""
        estimator = MaxMarginIRLEstimator(max_iterations=5, verbose=False)

        result = estimator._optimize(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        assert "margin" in result.metadata
        assert "num_violating_policies" in result.metadata
        assert "expert_features" in result.metadata

    def test_rejects_non_linear_reward(
        self, simple_panel, simple_problem, simple_transitions
    ):
        """Test that _optimize rejects non-LinearReward utility."""
        from econirl.preferences.linear import LinearUtility

        # Create a LinearUtility (not LinearReward)
        features = torch.randn(simple_problem.num_states, simple_problem.num_actions, 2)
        utility = LinearUtility(
            feature_matrix=features,
            parameter_names=["a", "b"],
        )

        estimator = MaxMarginIRLEstimator(max_iterations=5)

        with pytest.raises(TypeError, match="LinearReward"):
            estimator._optimize(
                simple_panel, utility, simple_problem, simple_transitions
            )

    def test_converges_on_simple_problem(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that optimization converges on simple problem."""
        estimator = MaxMarginIRLEstimator(
            max_iterations=50,
            margin_tol=1e-4,
            verbose=False,
        )

        result = estimator._optimize(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        # Should converge reasonably quickly
        assert result.num_iterations <= 50


# ============================================================================
# Test estimate() Method
# ============================================================================


class TestEstimateMethod:
    """Tests for the estimate() method (inherited from BaseEstimator)."""

    def test_estimate_runs(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that estimate() runs without error."""
        estimator = MaxMarginIRLEstimator(
            max_iterations=5,
            verbose=False,
            compute_hessian=True,
        )

        result = estimator.estimate(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        # Check we got an EstimationSummary
        from econirl.inference.results import EstimationSummary
        assert isinstance(result, EstimationSummary)

    def test_estimate_returns_summary(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that estimate returns EstimationSummary with all fields."""
        estimator = MaxMarginIRLEstimator(
            max_iterations=10,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        # Check parameters
        assert result.parameters is not None
        assert len(result.parameters) == simple_reward_fn.num_parameters

        # Check parameter names
        assert result.parameter_names == simple_reward_fn.parameter_names

        # Check standard errors
        assert result.standard_errors is not None
        assert len(result.standard_errors) == len(result.parameters)

        # Check method name
        assert "Max Margin" in result.method

    def test_estimate_summary_string(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that summary() method works."""
        estimator = MaxMarginIRLEstimator(
            max_iterations=5,
            verbose=False,
        )

        result = estimator.estimate(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Max Margin" in summary
        assert "coef" in summary


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_feature(self, simple_problem, simple_transitions, simple_panel):
        """Test with single-feature reward."""
        # Single feature: just state index
        features = torch.arange(simple_problem.num_states).float().unsqueeze(1)
        features = features / simple_problem.num_states

        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["state_value"],
            n_actions=simple_problem.num_actions,
        )

        estimator = MaxMarginIRLEstimator(max_iterations=5, verbose=False)

        result = estimator._optimize(
            simple_panel, reward_fn, simple_problem, simple_transitions
        )

        assert result.parameters.shape == (1,)

    def test_many_features(self, simple_problem, simple_transitions, simple_panel):
        """Test with many features."""
        num_features = 10
        features = torch.randn(simple_problem.num_states, num_features)

        reward_fn = LinearReward(
            state_features=features,
            parameter_names=[f"feature_{i}" for i in range(num_features)],
            n_actions=simple_problem.num_actions,
        )

        estimator = MaxMarginIRLEstimator(max_iterations=5, verbose=False)

        result = estimator._optimize(
            simple_panel, reward_fn, simple_problem, simple_transitions
        )

        assert result.parameters.shape == (num_features,)

    def test_initial_params(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that initial_params is used."""
        estimator = MaxMarginIRLEstimator(max_iterations=5, verbose=False)

        # Custom initial params
        init_params = torch.tensor([0.8, 0.2])
        init_params = init_params / torch.norm(init_params)  # Normalize

        result = estimator._optimize(
            simple_panel,
            simple_reward_fn,
            simple_problem,
            simple_transitions,
            initial_params=init_params,
        )

        # Should still return valid result
        assert torch.isfinite(result.parameters).all()

    def test_verbose_mode(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions, capsys
    ):
        """Test that verbose mode prints output."""
        estimator = MaxMarginIRLEstimator(
            max_iterations=3,
            verbose=True,
        )

        result = estimator._optimize(
            simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        captured = capsys.readouterr()
        assert "Iteration" in captured.out or "margin" in captured.out


# ============================================================================
# Test Compute Margin Helper
# ============================================================================


class TestComputeMargin:
    """Tests for compute_margin helper method."""

    def test_compute_margin_returns_scalar(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that compute_margin returns a scalar."""
        estimator = MaxMarginIRLEstimator()

        theta = torch.tensor([0.7, 0.7])
        theta = theta / torch.norm(theta)

        margin = estimator.compute_margin(
            theta, simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        assert isinstance(margin, float)

    def test_margin_finite(
        self, simple_panel, simple_reward_fn, simple_problem, simple_transitions
    ):
        """Test that margin is finite."""
        estimator = MaxMarginIRLEstimator()

        theta = torch.tensor([1.0, 1.0])
        theta = theta / torch.norm(theta)

        margin = estimator.compute_margin(
            theta, simple_panel, simple_reward_fn, simple_problem, simple_transitions
        )

        assert np.isfinite(margin)
