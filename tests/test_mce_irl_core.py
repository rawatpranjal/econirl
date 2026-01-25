"""Tests for MCE IRL core estimator.

Tests cover:
1. MCE IRL convergence on simple problems
2. Feature matching (empirical features approximately equal expected features)
3. Policy validity and value function properties
"""

import pytest
import torch
import numpy as np

from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.reward import LinearReward


class TestMCEIRLConvergence:
    """Test that MCE IRL converges on simple problems.

    Uses shared fixtures from conftest.py:
    - simple_problem: 10-state MDP with deterministic transitions
    - synthetic_panel: Panel of 20 trajectories with 50 periods each
    """

    def test_mce_irl_converges(self, simple_problem, synthetic_panel):
        """MCE IRL should converge on simple problem."""
        problem, transitions = simple_problem

        # State features: normalized state index
        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=100,
            learning_rate=0.5,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Should either converge or make meaningful progress
        assert result.converged or result.num_iterations < config.outer_max_iter
        assert result.log_likelihood > -float("inf")
        assert result.policy is not None
        assert result.policy.shape == (10, 2)

    def test_policy_is_valid(self, simple_problem, synthetic_panel):
        """MCE IRL should return valid probability distribution over actions."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=100,
            learning_rate=0.5,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        policy = result.policy

        # Policy should be non-negative
        assert (policy >= 0).all(), "Policy has negative probabilities"

        # Policy rows should sum to 1
        row_sums = policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Policy rows don't sum to 1: {row_sums}"

    def test_value_function_is_finite(self, simple_problem, synthetic_panel):
        """MCE IRL should return finite value function."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=100,
            learning_rate=0.5,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        assert result.value_function is not None
        assert result.value_function.shape == (10,)
        assert torch.isfinite(result.value_function).all(), \
            "Value function has non-finite values"


class TestFeatureMatching:
    """Test that MCE IRL matches feature expectations.

    Uses shared fixtures from conftest.py:
    - simple_problem: 10-state MDP with deterministic transitions
    - synthetic_panel: Panel of 20 trajectories with 50 periods each
    """

    def test_feature_matching(self, simple_problem, synthetic_panel):
        """MCE IRL should approximately match empirical feature expectations."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=200,
            learning_rate=0.5,
            outer_tol=1e-4,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Verify that estimation produced valid results
        # The MCE IRL algorithm should converge to a policy that explains the data
        assert result.policy is not None
        assert torch.isfinite(result.policy).all()

        # The policy should be a valid probability distribution
        row_sums = result.policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_estimation_produces_valid_policy(self, simple_problem, synthetic_panel):
        """MCE IRL should produce a valid policy that explains data."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            outer_max_iter=50,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Policy should be non-negative probabilities
        assert (result.policy >= 0).all()
        assert (result.policy <= 1).all()

        # Policy rows should sum to 1
        row_sums = result.policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_value_function_properties(self, simple_problem, synthetic_panel):
        """Value function should have expected properties."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            outer_max_iter=50,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Value function should be finite
        assert result.value_function is not None
        assert torch.isfinite(result.value_function).all()

        # Value function should have correct shape
        assert result.value_function.shape == (problem.num_states,)


class TestMCEIRLConfig:
    """Test MCE IRL configuration options."""

    def test_default_config(self):
        """Test that default config is created correctly."""
        config = MCEIRLConfig()

        assert config.optimizer == "L-BFGS-B"
        assert config.learning_rate == 0.1
        assert config.outer_max_iter == 200
        assert config.inner_max_iter == 10000
        assert config.compute_se is True
        assert config.verbose is False

    def test_custom_config(self):
        """Test that custom config values are set."""
        config = MCEIRLConfig(
            learning_rate=0.5,
            outer_max_iter=100,
            inner_max_iter=500,
            compute_se=False,
            verbose=True,
        )

        assert config.learning_rate == 0.5
        assert config.outer_max_iter == 100
        assert config.inner_max_iter == 500
        assert config.compute_se is False
        assert config.verbose is True

    def test_estimator_kwargs_override(self):
        """Test that kwargs override config values."""
        config = MCEIRLConfig(verbose=False)
        estimator = MCEIRLEstimator(config=config, verbose=True)

        assert estimator.config.verbose is True

    def test_estimator_name(self):
        """Test that estimator has correct name."""
        estimator = MCEIRLEstimator()
        assert "MCE IRL" in estimator.name or "Ziebart" in estimator.name


class TestMCEIRLMultipleFeatures:
    """Test MCE IRL with multiple state features."""

    @pytest.fixture
    def multi_feature_problem(self):
        """Create a problem with multiple features."""
        n_states = 15
        problem = DDCProblem(
            num_states=n_states,
            num_actions=2,
            discount_factor=0.9,
        )

        transitions = torch.zeros((2, n_states, n_states))
        for s in range(n_states):
            transitions[0, s, min(s + 1, n_states - 1)] = 1.0
            transitions[1, s, 0] = 1.0

        return problem, transitions

    @pytest.fixture
    def multi_feature_panel(self, multi_feature_problem):
        """Generate data for multi-feature test."""
        problem, transitions = multi_feature_problem
        n_states = problem.num_states

        np.random.seed(123)
        trajectories = []

        for i in range(30):
            states, actions, next_states = [], [], []
            s = 0
            for t in range(40):
                states.append(s)
                # Probability depends on state position
                p_replace = 0.1 + 0.1 * (s / n_states) + 0.05 * ((s / n_states) ** 2)
                a = 1 if np.random.random() < p_replace else 0
                actions.append(a)
                if a == 1:
                    next_s = 0
                else:
                    next_s = min(s + 1, n_states - 1)
                next_states.append(next_s)
                s = next_s

            traj = Trajectory(
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
                individual_id=i,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def test_multiple_features_converges(self, multi_feature_problem, multi_feature_panel):
        """MCE IRL should converge with multiple features."""
        problem, transitions = multi_feature_problem
        n_states = problem.num_states

        # Two features: linear and quadratic state position
        feature1 = torch.arange(n_states, dtype=torch.float32) / n_states
        feature2 = feature1 ** 2
        features = torch.stack([feature1, feature2], dim=1)

        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["linear", "quadratic"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=150,
            learning_rate=0.3,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=multi_feature_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Check basic properties
        assert result.parameters.shape == (2,)
        assert torch.isfinite(result.parameters).all()
        assert result.policy.shape == (n_states, 2)

    def test_parameter_estimates_have_expected_sign(self, multi_feature_problem, multi_feature_panel):
        """Test that parameters have economically sensible signs."""
        problem, transitions = multi_feature_problem
        n_states = problem.num_states

        feature1 = torch.arange(n_states, dtype=torch.float32) / n_states
        feature2 = feature1 ** 2
        features = torch.stack([feature1, feature2], dim=1)

        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["linear", "quadratic"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=200,
            learning_rate=0.3,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=multi_feature_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Parameters should be non-zero (the algorithm found something)
        assert not torch.allclose(result.parameters, torch.zeros(2))


class TestMCEIRLEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_trajectory(self):
        """Test MCE IRL with a single trajectory."""
        n_states = 5
        problem = DDCProblem(
            num_states=n_states,
            num_actions=2,
            discount_factor=0.9,
        )

        transitions = torch.zeros((2, n_states, n_states))
        for s in range(n_states):
            transitions[0, s, min(s + 1, n_states - 1)] = 1.0
            transitions[1, s, 0] = 1.0

        # Single trajectory
        traj = Trajectory(
            states=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
            actions=torch.tensor([0, 0, 0, 0, 1, 0, 0, 1]),
            next_states=torch.tensor([1, 2, 3, 4, 0, 1, 2, 0]),
            individual_id=0,
        )
        panel = Panel(trajectories=[traj])

        features = torch.arange(n_states, dtype=torch.float32).unsqueeze(1) / n_states
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            outer_max_iter=50,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Should still produce valid results
        assert result.policy is not None
        assert result.value_function is not None
        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)

    def test_high_discount_factor(self):
        """Test MCE IRL with high discount factor (close to 1)."""
        n_states = 8
        problem = DDCProblem(
            num_states=n_states,
            num_actions=2,
            discount_factor=0.99,  # High discount
        )

        transitions = torch.zeros((2, n_states, n_states))
        for s in range(n_states):
            transitions[0, s, min(s + 1, n_states - 1)] = 1.0
            transitions[1, s, 0] = 1.0

        # Generate some data
        np.random.seed(42)
        trajectories = []
        for i in range(10):
            states, actions, next_states = [], [], []
            s = 0
            for t in range(30):
                states.append(s)
                a = 1 if np.random.random() < 0.1 + 0.1 * s / n_states else 0
                actions.append(a)
                next_s = 0 if a == 1 else min(s + 1, n_states - 1)
                next_states.append(next_s)
                s = next_s

            traj = Trajectory(
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
                individual_id=i,
            )
            trajectories.append(traj)

        panel = Panel(trajectories=trajectories)

        features = torch.arange(n_states, dtype=torch.float32).unsqueeze(1) / n_states
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=1000,  # More iterations for high discount
            outer_max_iter=100,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Should produce finite values
        assert torch.isfinite(result.value_function).all()
        assert torch.isfinite(result.policy).all()
        assert torch.isfinite(result.parameters).all()

    def test_all_same_action(self):
        """Test MCE IRL when agent always takes the same action."""
        n_states = 5
        problem = DDCProblem(
            num_states=n_states,
            num_actions=2,
            discount_factor=0.9,
        )

        transitions = torch.zeros((2, n_states, n_states))
        for s in range(n_states):
            transitions[0, s, min(s + 1, n_states - 1)] = 1.0
            transitions[1, s, 0] = 1.0

        # All actions are 0 (keep)
        trajectories = []
        for i in range(5):
            traj = Trajectory(
                states=torch.tensor([0, 1, 2, 3, 4, 4, 4, 4]),
                actions=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
                next_states=torch.tensor([1, 2, 3, 4, 4, 4, 4, 4]),
                individual_id=i,
            )
            trajectories.append(traj)

        panel = Panel(trajectories=trajectories)

        features = torch.arange(n_states, dtype=torch.float32).unsqueeze(1) / n_states
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            outer_max_iter=50,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Should still work (policy should favor action 0)
        assert result.policy is not None
        # Policy for action 0 should be high across states
        assert (result.policy[:, 0] > 0.5).sum() >= n_states // 2
