"""Unit tests for Guided Cost Learning (GCL).

Tests for:
- NeuralCostFunction (embeddings, MLP, compute)
- GCLEstimator helper methods (trajectory cost, importance weights, sampling)
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.contrib.gcl import GCLEstimator, GCLConfig
from econirl.preferences.neural_cost import NeuralCostFunction


class TestNeuralCostFunction:
    """Tests for NeuralCostFunction."""

    def test_neural_cost_forward_batch(self):
        """Test forward pass with batched state-action pairs."""
        cost_fn = NeuralCostFunction(n_states=10, n_actions=2, embed_dim=8, hidden_dims=[16])

        states = jnp.array([0, 1, 2, 3, 4])
        actions = jnp.array([0, 1, 0, 1, 0])
        costs = cost_fn.forward(states, actions)

        assert costs.shape == (5,), f"Expected shape (5,), got {costs.shape}"
        assert jnp.all(jnp.isfinite(costs)), "All costs should be finite"

    def test_neural_cost_compute_shape(self):
        """Test compute() returns correct shape matrix."""
        n_states, n_actions = 20, 3
        cost_fn = NeuralCostFunction(n_states=n_states, n_actions=n_actions, embed_dim=16, hidden_dims=[32])

        cost_matrix = cost_fn.compute()

        assert cost_matrix.shape == (n_states, n_actions), f"Expected ({n_states}, {n_actions}), got {cost_matrix.shape}"
        assert jnp.all(jnp.isfinite(cost_matrix)), "All costs should be finite"

    def test_neural_cost_compute_consistency(self):
        """Test that compute() matches forward() for all state-action pairs."""
        n_states, n_actions = 5, 2
        cost_fn = NeuralCostFunction(n_states=n_states, n_actions=n_actions, embed_dim=8, hidden_dims=[16])

        cost_matrix = cost_fn.compute()

        # Check each entry matches forward()
        for s in range(n_states):
            for a in range(n_actions):
                expected = cost_fn.forward(jnp.array([s]), jnp.array([a]))[0]
                actual = cost_matrix[s, a]
                assert jnp.allclose(expected, actual, atol=1e-6), f"Mismatch at ({s}, {a})"

    def test_neural_cost_reward_matrix(self):
        """Test that reward matrix is negative of cost matrix."""
        cost_fn = NeuralCostFunction(n_states=10, n_actions=2)

        cost_matrix = cost_fn.compute()
        reward_matrix = cost_fn.get_reward_matrix()

        assert jnp.allclose(reward_matrix, -cost_matrix), "Reward should be negative cost"

    def test_neural_cost_different_architectures(self):
        """Test cost function with different architectures."""
        # Single hidden layer
        cost_fn1 = NeuralCostFunction(n_states=10, n_actions=2, embed_dim=4, hidden_dims=[8])
        assert cost_fn1.compute().shape == (10, 2)

        # Multiple hidden layers
        cost_fn2 = NeuralCostFunction(n_states=10, n_actions=2, embed_dim=16, hidden_dims=[32, 32, 16])
        assert cost_fn2.compute().shape == (10, 2)

        # Different embedding dimension
        cost_fn3 = NeuralCostFunction(n_states=10, n_actions=2, embed_dim=64, hidden_dims=[128])
        assert cost_fn3.compute().shape == (10, 2)

    def test_neural_cost_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "tanh", "leaky_relu"]:
            cost_fn = NeuralCostFunction(
                n_states=5, n_actions=2, embed_dim=8, hidden_dims=[16], activation=activation
            )
            cost_matrix = cost_fn.compute()
            assert cost_matrix.shape == (5, 2)
            assert jnp.all(jnp.isfinite(cost_matrix))

    def test_neural_cost_repr(self):
        """Test string representation."""
        cost_fn = NeuralCostFunction(n_states=10, n_actions=2, embed_dim=32, hidden_dims=[64, 64])
        repr_str = repr(cost_fn)

        assert "NeuralCostFunction" in repr_str
        assert "n_states=10" in repr_str
        assert "n_actions=2" in repr_str
        assert "embed_dim=32" in repr_str


class TestGCLEstimatorHelpers:
    """Tests for GCLEstimator helper methods."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup."""
        n_states, n_actions = 5, 2

        # Create simple transitions
        transitions = jnp.zeros((n_actions, n_states, n_states))
        for s in range(n_states):
            if s < n_states - 1:
                transitions = transitions.at[0, s, s + 1].set(1.0)  # keep -> advance
            else:
                transitions = transitions.at[0, s, s].set(1.0)  # stay at terminal
            transitions = transitions.at[1, s, 0].set(1.0)  # replace -> go to 0

        # Create panel
        trajectories = [
            Trajectory(
                states=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
                actions=jnp.array([0, 0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 3, 0], dtype=jnp.int32),
            ),
            Trajectory(
                states=jnp.array([0, 1, 2], dtype=jnp.int32),
                actions=jnp.array([0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
            ),
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
            max_iterations=5,
            n_sample_trajectories=10,
            trajectory_length=5,
        )
        estimator = GCLEstimator(config=config)

        return {
            "n_states": n_states,
            "n_actions": n_actions,
            "transitions": transitions,
            "panel": panel,
            "problem": problem,
            "estimator": estimator,
        }

    def test_trajectory_cost_computation(self, simple_setup):
        """Test that trajectory cost sums individual costs correctly."""
        estimator = simple_setup["estimator"]
        n_states = simple_setup["n_states"]
        n_actions = simple_setup["n_actions"]

        cost_fn = NeuralCostFunction(n_states=n_states, n_actions=n_actions, embed_dim=8, hidden_dims=[16])

        # Create a trajectory
        traj = Trajectory(
            states=jnp.array([0, 1, 2], dtype=jnp.int32),
            actions=jnp.array([0, 1, 0], dtype=jnp.int32),
            next_states=jnp.array([1, 2, 3], dtype=jnp.int32),
        )

        # Compute trajectory cost
        total_cost = estimator._compute_trajectory_cost(cost_fn, traj)

        # Manually compute expected cost
        expected_cost = cost_fn.forward(traj.states, traj.actions).sum()

        assert jnp.allclose(total_cost, expected_cost, atol=1e-6), "Trajectory cost should sum individual costs"

    def test_trajectory_log_prob(self, simple_setup):
        """Test trajectory log probability computation."""
        estimator = simple_setup["estimator"]
        n_states = simple_setup["n_states"]
        n_actions = simple_setup["n_actions"]

        # Create a deterministic policy for testing
        policy = jnp.zeros((n_states, n_actions))
        policy = policy.at[:, 0].set(0.8)
        policy = policy.at[:, 1].set(0.2)

        traj = Trajectory(
            states=jnp.array([0, 1, 2], dtype=jnp.int32),
            actions=jnp.array([0, 0, 1], dtype=jnp.int32),
            next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
        )

        log_prob = estimator._compute_trajectory_log_prob(policy, traj)

        # Expected: log(0.8) + log(0.8) + log(0.2)
        expected = np.log(0.8) + np.log(0.8) + np.log(0.2)

        assert jnp.isclose(log_prob, jnp.float32(expected), atol=1e-5), "Log prob should be sum of log action probs"

    def test_importance_weights_normalized(self, simple_setup):
        """Test that importance weights are normalized."""
        estimator = simple_setup["estimator"]
        n_states = simple_setup["n_states"]
        n_actions = simple_setup["n_actions"]

        cost_fn = NeuralCostFunction(n_states=n_states, n_actions=n_actions, embed_dim=8, hidden_dims=[16])
        policy = jnp.ones((n_states, n_actions)) / n_actions

        # Create some trajectories
        trajectories = [
            Trajectory(
                states=jnp.array([0, 1, 2], dtype=jnp.int32),
                actions=jnp.array([0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
            ),
            Trajectory(
                states=jnp.array([0, 1], dtype=jnp.int32),
                actions=jnp.array([1, 0], dtype=jnp.int32),
                next_states=jnp.array([0, 2], dtype=jnp.int32),
            ),
            Trajectory(
                states=jnp.array([2, 3, 4], dtype=jnp.int32),
                actions=jnp.array([0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([3, 4, 0], dtype=jnp.int32),
            ),
        ]

        weights = estimator._compute_importance_weights(cost_fn, policy, trajectories)

        # Weights should sum to 1
        assert jnp.isclose(weights.sum(), 1.0, atol=1e-6), "Weights should sum to 1"

        # All weights should be non-negative
        assert jnp.all(weights >= 0), "Weights should be non-negative"

    def test_importance_weights_clipping(self, simple_setup):
        """Test that importance weights are properly clipped."""
        n_states = simple_setup["n_states"]
        n_actions = simple_setup["n_actions"]

        # Create estimator with low clipping threshold
        config = GCLConfig(
            embed_dim=8,
            hidden_dims=[16],
            importance_clipping=2.0,  # Low clipping
        )
        estimator = GCLEstimator(config=config)

        cost_fn = NeuralCostFunction(n_states=n_states, n_actions=n_actions, embed_dim=8, hidden_dims=[16])
        policy = jnp.ones((n_states, n_actions)) / n_actions

        trajectories = [
            Trajectory(
                states=jnp.array([0, 1], dtype=jnp.int32),
                actions=jnp.array([0, 0], dtype=jnp.int32),
                next_states=jnp.array([1, 2], dtype=jnp.int32),
            )
            for _ in range(10)
        ]

        weights = estimator._compute_importance_weights(cost_fn, policy, trajectories)

        # Max weight should be clipped
        max_weight = config.importance_clipping / len(trajectories)
        assert jnp.all(weights <= max_weight + 1e-6), f"Weights should be clipped to {max_weight}"

    def test_trajectory_sampling(self, simple_setup):
        """Test that sampled trajectories are valid."""
        estimator = simple_setup["estimator"]
        transitions = simple_setup["transitions"]
        n_states = simple_setup["n_states"]
        n_actions = simple_setup["n_actions"]

        policy = jnp.ones((n_states, n_actions)) / n_actions
        initial_dist = jnp.ones(n_states) / n_states

        trajectories = estimator._sample_trajectories(
            policy=policy,
            transitions=transitions,
            initial_dist=initial_dist,
            n_trajectories=20,
            trajectory_length=10,
        )

        assert len(trajectories) == 20, "Should sample correct number of trajectories"

        for traj in trajectories:
            assert len(traj) == 10, "Each trajectory should have correct length"
            assert jnp.all(traj.states >= 0) and jnp.all(traj.states < n_states), "States should be valid"
            assert jnp.all(traj.actions >= 0) and jnp.all(traj.actions < n_actions), "Actions should be valid"

    def test_trajectory_sampling_respects_transitions(self, simple_setup):
        """Test that sampled trajectories follow transition dynamics."""
        estimator = simple_setup["estimator"]
        transitions = simple_setup["transitions"]
        n_states = simple_setup["n_states"]
        n_actions = simple_setup["n_actions"]

        # Create deterministic policy (always action 0)
        policy = jnp.zeros((n_states, n_actions))
        policy = policy.at[:, 0].set(1.0)

        # Start from state 0
        initial_dist = jnp.zeros(n_states)
        initial_dist = initial_dist.at[0].set(1.0)

        trajectories = estimator._sample_trajectories(
            policy=policy,
            transitions=transitions,
            initial_dist=initial_dist,
            n_trajectories=5,
            trajectory_length=4,
        )

        for traj in trajectories:
            # With deterministic policy action=0 and our transitions,
            # should see states 0 -> 1 -> 2 -> 3
            assert int(traj.states[0]) == 0, "Should start at state 0"
            # Check transition consistency
            for t in range(len(traj) - 1):
                s = int(traj.states[t])
                a = int(traj.actions[t])
                s_next = int(traj.next_states[t])
                # Check that this transition has non-zero probability
                assert float(transitions[a, s, s_next]) > 0, f"Invalid transition from {s} with action {a} to {s_next}"

    def test_initial_distribution_from_panel(self, simple_setup):
        """Test computing initial state distribution from panel."""
        estimator = simple_setup["estimator"]
        panel = simple_setup["panel"]
        n_states = simple_setup["n_states"]

        initial_dist = estimator._compute_initial_distribution(panel, n_states)

        # Should sum to 1
        assert jnp.isclose(initial_dist.sum(), 1.0), "Initial dist should sum to 1"

        # State 0 should have all the mass (both trajectories start there)
        assert float(initial_dist[0]) == 1.0, "All trajectories start at state 0"


class TestGCLEstimatorOptimization:
    """Tests for GCLEstimator optimization."""

    def test_gcl_runs_without_error(self):
        """Test that GCL optimization runs without errors."""
        n_states, n_actions = 5, 2

        trajectories = [
            Trajectory(
                states=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
                actions=jnp.array([0, 0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 3, 0], dtype=jnp.int32),
            ),
            Trajectory(
                states=jnp.array([0, 1, 2], dtype=jnp.int32),
                actions=jnp.array([0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
            ),
        ]
        panel = Panel(trajectories=trajectories)

        transitions = jnp.zeros((n_actions, n_states, n_states))
        for s in range(n_states):
            if s < n_states - 1:
                transitions = transitions.at[0, s, s + 1].set(1.0)
            else:
                transitions = transitions.at[0, s, s].set(1.0)
            transitions = transitions.at[1, s, 0].set(1.0)

        problem = DDCProblem(num_states=n_states, num_actions=n_actions, discount_factor=0.9)

        config = GCLConfig(
            embed_dim=8,
            hidden_dims=[16],
            max_iterations=3,
            n_sample_trajectories=5,
            trajectory_length=3,
        )
        estimator = GCLEstimator(config=config)

        dummy_utility = NeuralCostFunction(n_states, n_actions, embed_dim=8, hidden_dims=[16])
        result = estimator.estimate(panel, dummy_utility, problem, transitions)

        assert result.policy.shape == (n_states, n_actions)
        assert result.value_function.shape == (n_states,)
        assert np.isfinite(result.log_likelihood)

    def test_gcl_stores_cost_function(self):
        """Test that GCL stores the learned cost function."""
        n_states, n_actions = 5, 2

        trajectories = [
            Trajectory(
                states=jnp.array([0, 1, 2], dtype=jnp.int32),
                actions=jnp.array([0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 0], dtype=jnp.int32),
            ),
        ]
        panel = Panel(trajectories=trajectories)

        transitions = jnp.zeros((n_actions, n_states, n_states))
        for s in range(n_states):
            if s < n_states - 1:
                transitions = transitions.at[0, s, s + 1].set(1.0)
            else:
                transitions = transitions.at[0, s, s].set(1.0)
            transitions = transitions.at[1, s, 0].set(1.0)

        problem = DDCProblem(num_states=n_states, num_actions=n_actions, discount_factor=0.9)

        config = GCLConfig(embed_dim=8, hidden_dims=[16], max_iterations=2)
        estimator = GCLEstimator(config=config)

        dummy_utility = NeuralCostFunction(n_states, n_actions)
        estimator.estimate(panel, dummy_utility, problem, transitions)

        assert estimator.cost_function_ is not None
        assert isinstance(estimator.cost_function_, NeuralCostFunction)

    def test_gcl_policy_is_valid_distribution(self):
        """Test that GCL produces valid probability distributions."""
        n_states, n_actions = 5, 2

        trajectories = [
            Trajectory(
                states=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
                actions=jnp.array([0, 0, 0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 2, 3, 0], dtype=jnp.int32),
            ),
        ]
        panel = Panel(trajectories=trajectories)

        transitions = jnp.zeros((n_actions, n_states, n_states))
        for s in range(n_states):
            if s < n_states - 1:
                transitions = transitions.at[0, s, s + 1].set(1.0)
            else:
                transitions = transitions.at[0, s, s].set(1.0)
            transitions = transitions.at[1, s, 0].set(1.0)

        problem = DDCProblem(num_states=n_states, num_actions=n_actions, discount_factor=0.9)

        config = GCLConfig(embed_dim=8, hidden_dims=[16], max_iterations=3)
        estimator = GCLEstimator(config=config)

        dummy_utility = NeuralCostFunction(n_states, n_actions)
        result = estimator.estimate(panel, dummy_utility, problem, transitions)

        # Policy should sum to 1 for each state
        policy_sums = result.policy.sum(axis=1)
        assert jnp.allclose(policy_sums, jnp.ones(n_states), atol=1e-6), "Policy should be valid distribution"

        # Policy should be non-negative
        assert jnp.all(result.policy >= 0), "Policy probabilities should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
