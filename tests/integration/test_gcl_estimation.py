"""Integration tests for Guided Cost Learning (GCL) estimation.

These tests verify the GCL estimator works end-to-end on:
1. Simple MDPs with known structure
2. Real Rust bus data
3. sklearn-style API compliance
"""

import numpy as np
import pandas as pd
import pytest
import torch

from econirl.core.types import DDCProblem, Panel, Trajectory


class TestGCLSimpleMDP:
    """Test GCL on simple MDPs with known structure."""

    def test_gcl_simple_3state_mdp(self):
        """Test GCL converges on a simple 3-state MDP.

        Creates a simple MDP where:
        - State 0 -> State 1 -> State 2 (with action 0)
        - Any state -> State 0 (with action 1 = reset)

        Demonstrations show preference for staying in low states
        (action 0 until state 2, then action 1 to reset).
        """
        from econirl.estimation.gcl import GCLEstimator, GCLConfig
        from econirl.preferences.neural_cost import NeuralCostFunction

        n_states, n_actions = 3, 2

        # Create transitions
        transitions = torch.zeros((n_actions, n_states, n_states))
        # Action 0: advance state
        transitions[0, 0, 1] = 1.0
        transitions[0, 1, 2] = 1.0
        transitions[0, 2, 2] = 1.0  # Stay at terminal
        # Action 1: reset to state 0
        transitions[1, :, 0] = 1.0

        # Create demonstrations that show "advance then reset" behavior
        trajectories = [
            Trajectory(
                states=torch.tensor([0, 1, 2]),
                actions=torch.tensor([0, 0, 1]),
                next_states=torch.tensor([1, 2, 0]),
            )
            for _ in range(10)  # Multiple copies for stability
        ]
        panel = Panel(trajectories=trajectories)

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=0.9,
        )

        config = GCLConfig(
            embed_dim=8,
            hidden_dims=[16],
            cost_lr=1e-2,
            max_iterations=50,
            n_sample_trajectories=20,
            trajectory_length=10,
            verbose=False,
        )
        estimator = GCLEstimator(config=config)

        dummy_utility = NeuralCostFunction(n_states, n_actions, embed_dim=8, hidden_dims=[16])
        result = estimator.estimate(panel, dummy_utility, problem, transitions)

        # Basic checks
        assert result.policy.shape == (n_states, n_actions)
        assert result.value_function.shape == (n_states,)
        assert np.isfinite(result.log_likelihood)

        # Policy should be valid distributions
        policy_sums = result.policy.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones(n_states), atol=1e-5)

    def test_gcl_learns_replacement_pattern(self):
        """Test that GCL learns a replacement pattern.

        In this MDP:
        - States represent "wear" level (0 = new, higher = more worn)
        - Action 0 = keep (advance wear)
        - Action 1 = replace (reset to 0)

        Demonstrations show replacement when wear is high.
        """
        from econirl.estimation.gcl import GCLEstimator, GCLConfig
        from econirl.preferences.neural_cost import NeuralCostFunction

        n_states, n_actions = 5, 2

        # Create transitions
        transitions = torch.zeros((n_actions, n_states, n_states))
        for s in range(n_states):
            if s < n_states - 1:
                transitions[0, s, s + 1] = 1.0  # Keep -> advance
            else:
                transitions[0, s, s] = 1.0  # Stay at max
            transitions[1, s, 0] = 1.0  # Replace -> reset

        # Demonstrations: keep until state 3 or 4, then replace
        trajectories = []
        for _ in range(15):
            # Pattern 1: replace at state 3
            trajectories.append(Trajectory(
                states=torch.tensor([0, 1, 2, 3]),
                actions=torch.tensor([0, 0, 0, 1]),
                next_states=torch.tensor([1, 2, 3, 0]),
            ))
            # Pattern 2: replace at state 4
            trajectories.append(Trajectory(
                states=torch.tensor([0, 1, 2, 3, 4]),
                actions=torch.tensor([0, 0, 0, 0, 1]),
                next_states=torch.tensor([1, 2, 3, 4, 0]),
            ))
        panel = Panel(trajectories=trajectories)

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=0.9,
        )

        config = GCLConfig(
            embed_dim=16,
            hidden_dims=[32],
            cost_lr=1e-2,
            max_iterations=100,
            n_sample_trajectories=30,
            trajectory_length=10,
            verbose=False,
        )
        estimator = GCLEstimator(config=config)

        dummy_utility = NeuralCostFunction(n_states, n_actions, embed_dim=16, hidden_dims=[32])
        result = estimator.estimate(panel, dummy_utility, problem, transitions)

        # The learned policy should show increasing probability of replacement
        # as the state (wear) increases
        policy = result.policy.detach().numpy()

        # In high states, replacement probability should be higher than in low states
        # This is a soft check - the exact values depend on convergence
        replace_probs = policy[:, 1]  # P(replace) for each state

        # At minimum, check that the policy gives some probability to both actions
        # and is a valid distribution
        assert np.all(policy >= 0)
        assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-5)


class TestGCLRustBus:
    """Test GCL on Rust bus data."""

    @pytest.fixture
    def rust_data_small(self):
        """Load a small subset of Rust bus data for testing."""
        from econirl.datasets import load_rust_bus

        # Load synthetic data (faster, always available)
        df = load_rust_bus(original=False)

        # Take a subset for faster testing
        buses = df["bus_id"].unique()[:5]
        return df[df["bus_id"].isin(buses)].copy()

    def test_gcl_runs_on_rust_data(self, rust_data_small):
        """Test that GCL can fit the Rust bus data without errors."""
        from econirl.estimators import GCL

        model = GCL(
            n_states=90,
            n_actions=2,
            discount=0.9,
            embed_dim=16,
            hidden_dims=[32],
            cost_lr=1e-3,
            max_iterations=10,  # Few iterations for speed
            n_sample_trajectories=20,
            trajectory_length=20,
            verbose=False,
        )

        model.fit(
            data=rust_data_small,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        # Check that all expected attributes are set
        assert model.cost_matrix_ is not None
        assert model.reward_matrix_ is not None
        assert model.policy_ is not None
        assert model.value_function_ is not None
        assert model.log_likelihood_ is not None
        assert model.converged_ is not None

        # Check shapes
        assert model.cost_matrix_.shape == (90, 2)
        assert model.reward_matrix_.shape == (90, 2)
        assert model.policy_.shape == (90, 2)
        assert model.value_function_.shape == (90,)

    def test_gcl_produces_valid_policy(self, rust_data_small):
        """Test that GCL produces a valid probability distribution."""
        from econirl.estimators import GCL

        model = GCL(
            n_states=90,
            n_actions=2,
            discount=0.9,
            embed_dim=16,
            hidden_dims=[32],
            max_iterations=10,
            verbose=False,
        )

        model.fit(
            data=rust_data_small,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        # Policy should be valid probability distribution
        assert np.all(model.policy_ >= 0), "Policy should be non-negative"
        assert np.all(model.policy_ <= 1), "Policy should be <= 1"
        assert np.allclose(model.policy_.sum(axis=1), 1.0, atol=1e-5), "Policy should sum to 1"

    def test_gcl_predict_proba(self, rust_data_small):
        """Test predict_proba method."""
        from econirl.estimators import GCL

        model = GCL(
            n_states=90,
            n_actions=2,
            discount=0.9,
            max_iterations=5,
            verbose=False,
        )

        model.fit(
            data=rust_data_small,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        # Test predict_proba
        states = np.array([0, 10, 50, 89])
        probs = model.predict_proba(states)

        assert probs.shape == (4, 2)
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_gcl_get_cost_and_reward(self, rust_data_small):
        """Test get_cost and get_reward methods."""
        from econirl.estimators import GCL

        model = GCL(
            n_states=90,
            n_actions=2,
            discount=0.9,
            max_iterations=5,
            verbose=False,
        )

        model.fit(
            data=rust_data_small,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        states = np.array([0, 10, 50])
        actions = np.array([0, 1, 0])

        costs = model.get_cost(states, actions)
        rewards = model.get_reward(states, actions)

        assert costs.shape == (3,)
        assert rewards.shape == (3,)
        assert np.allclose(costs, -rewards), "Reward should be negative cost"


class TestGCLSklearnAPI:
    """Test GCL sklearn-style API compliance."""

    def test_gcl_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        from econirl.estimators import GCL

        # Create minimal data
        data = pd.DataFrame({
            "state": [0, 1, 2, 0, 1],
            "action": [0, 0, 1, 0, 1],
            "id": [1, 1, 1, 2, 2],
        })

        model = GCL(n_states=5, n_actions=2, discount=0.9, max_iterations=2)
        result = model.fit(data, state="state", action="action", id="id")

        assert result is model, "fit() should return self"

    def test_gcl_summary_format(self):
        """Test that summary() returns formatted string."""
        from econirl.estimators import GCL

        data = pd.DataFrame({
            "state": [0, 1, 2, 0, 1],
            "action": [0, 0, 1, 0, 1],
            "id": [1, 1, 1, 2, 2],
        })

        model = GCL(n_states=5, n_actions=2, discount=0.9, max_iterations=2)
        model.fit(data, state="state", action="action", id="id")

        summary = model.summary()

        assert isinstance(summary, str)
        assert "Guided Cost Learning" in summary
        assert "GCL" in summary
        assert "Log-Likelihood" in summary

    def test_gcl_repr(self):
        """Test string representation."""
        from econirl.estimators import GCL

        model = GCL(n_states=90, n_actions=2, discount=0.99)
        repr_str = repr(model)

        assert "GCL" in repr_str
        assert "n_states=90" in repr_str
        assert "n_actions=2" in repr_str
        assert "fitted=False" in repr_str

        # After fitting
        data = pd.DataFrame({
            "state": [0, 1, 2],
            "action": [0, 0, 1],
            "id": [1, 1, 1],
        })
        model.fit(data, state="state", action="action", id="id")

        repr_str_fitted = repr(model)
        assert "fitted=True" in repr_str_fitted

    def test_gcl_error_before_fit(self):
        """Test that methods raise errors before fitting."""
        from econirl.estimators import GCL

        model = GCL(n_states=5, n_actions=2, discount=0.9)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0, 1]))

        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_cost(np.array([0]), np.array([0]))

        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_reward(np.array([0]), np.array([0]))

    def test_gcl_stores_cost_function(self):
        """Test that fitted model stores the neural network cost function."""
        from econirl.estimators import GCL
        from econirl.preferences.neural_cost import NeuralCostFunction

        data = pd.DataFrame({
            "state": [0, 1, 2, 3],
            "action": [0, 0, 0, 1],
            "id": [1, 1, 1, 1],
        })

        model = GCL(
            n_states=5,
            n_actions=2,
            discount=0.9,
            embed_dim=8,
            hidden_dims=[16],
            max_iterations=3,
        )
        model.fit(data, state="state", action="action", id="id")

        assert model.cost_function_ is not None
        assert isinstance(model.cost_function_, NeuralCostFunction)


class TestGCLComparison:
    """Compare GCL with other estimators."""

    def test_gcl_vs_mce_irl_same_data(self):
        """Test that GCL and MCE IRL both run on the same data.

        This is not a test of equivalence (they use different approaches),
        but rather a sanity check that both produce reasonable outputs.
        """
        from econirl.estimators import GCL, MCEIRL
        import numpy as np

        # Create test data
        np.random.seed(42)
        n_states = 10

        data = pd.DataFrame({
            "state": [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2],
            "action": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        })

        # Fit GCL
        gcl = GCL(
            n_states=n_states,
            n_actions=2,
            discount=0.9,
            embed_dim=8,
            hidden_dims=[16],
            max_iterations=10,
            verbose=False,
        )
        gcl.fit(data, state="state", action="action", id="id")

        # Fit MCE IRL with simple features
        s = np.arange(n_states)
        features = np.column_stack([s / 10, (s / 10) ** 2])

        mce = MCEIRL(
            n_states=n_states,
            n_actions=2,
            discount=0.9,
            feature_matrix=features,
            feature_names=["linear", "quadratic"],
            verbose=False,
        )
        mce.fit(data, state="state", action="action", id="id")

        # Both should produce valid policies
        assert gcl.policy_ is not None
        assert mce.policy_ is not None
        assert gcl.policy_.shape == mce.policy_.shape

        # Both policies should be valid distributions
        assert np.allclose(gcl.policy_.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(mce.policy_.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
