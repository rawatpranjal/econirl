"""Tests for IQ-Learn estimator."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
from econirl.preferences.action_reward import ActionDependentReward


@pytest.fixture
def simple_problem():
    """Create a simple 3-state, 2-action problem."""
    return DDCProblem(
        num_states=3,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
    )


@pytest.fixture
def simple_transitions(simple_problem):
    """Create simple deterministic transitions."""
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions
    transitions = jnp.zeros((n_actions, n_states, n_states))

    for s in range(n_states):
        transitions = transitions.at[0, s, s].set(1.0)
    for s in range(n_states):
        next_s = (s + 1) % n_states
        transitions = transitions.at[1, s, next_s].set(1.0)

    return transitions


@pytest.fixture
def simple_reward_fn(simple_problem):
    """Create simple action-dependent reward function."""
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions
    features = jnp.zeros((n_states, n_actions, 2))
    features = features.at[:, 0, 0].set(1.0)
    features = features.at[:, 1, 1].set(1.0)
    return ActionDependentReward(
        feature_matrix=features,
        parameter_names=["action_0_reward", "action_1_reward"],
    )


@pytest.fixture
def expert_panel():
    """Create expert demonstrations favoring action 0."""
    trajectories = []
    for i in range(20):
        states = jnp.array([0, 0, 0, 0, 0])
        actions = jnp.array([0, 0, 0, 0, 0])
        next_states = jnp.array([0, 0, 0, 0, 0])
        trajectories.append(Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=i,
        ))
    return Panel(trajectories=trajectories)


class TestIQLearnEstimator:
    """Tests for IQ-Learn estimator."""

    def test_iq_learn_init(self):
        """IQ-Learn should initialize with default config."""
        estimator = IQLearnEstimator()
        assert estimator.name == "IQ-Learn (Garg et al. 2021)"

    def test_iq_learn_init_with_config(self):
        """IQ-Learn should accept custom config."""
        config = IQLearnConfig(q_type="linear", alpha=0.5)
        estimator = IQLearnEstimator(config=config)
        assert estimator.config.q_type == "linear"
        assert estimator.config.alpha == 0.5

    def test_iq_learn_estimate_returns_result(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """IQ-Learn estimate should return EstimationSummary."""
        config = IQLearnConfig(max_iter=50, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert result.parameters is not None
        assert result.policy is not None
        assert result.policy.shape == (3, 2)

    def test_policy_is_valid_distribution(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Policy rows should sum to 1."""
        config = IQLearnConfig(max_iter=50, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        policy_sum = result.policy.sum(axis=1)
        assert jnp.allclose(policy_sum, jnp.ones(3), atol=1e-5)

    def test_recovers_reward_structure(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """IQ-Learn should recover relative reward structure from expert data."""
        config = IQLearnConfig(max_iter=200, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Expert always takes action 0, so learned policy should prefer action 0
        assert result.policy[0, 0] > result.policy[0, 1]

    def test_linear_q_type(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Linear Q mode should return feature-weight parameters."""
        config = IQLearnConfig(q_type="linear", max_iter=100, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Linear mode should return same number of params as features
        assert len(result.parameters) == simple_reward_fn.num_parameters
        assert result.parameter_names == simple_reward_fn.parameter_names

    def test_simple_divergence(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Simple divergence should also converge."""
        config = IQLearnConfig(divergence="simple", max_iter=100, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert result.policy is not None
        policy_sum = result.policy.sum(axis=1)
        assert jnp.allclose(policy_sum, jnp.ones(3), atol=1e-5)

    def test_metadata_contains_reward_table(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Metadata should include Q-table and reward table."""
        config = IQLearnConfig(max_iter=50, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert "q_table" in result.metadata
        assert "reward_table" in result.metadata
        assert "divergence" in result.metadata

    def test_reward_recovery_bellman_consistency(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Recovered reward should satisfy r = Q - gamma * E[V*(s')]."""
        config = IQLearnConfig(max_iter=100, verbose=False)
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        Q_table = jnp.array(result.metadata["q_table"])
        reward_table = jnp.array(result.metadata["reward_table"])
        gamma = simple_problem.discount_factor
        sigma = simple_problem.scale_parameter

        V = sigma * jax.scipy.special.logsumexp(Q_table / sigma, axis=1)
        EV = jnp.einsum("ast,t->as", simple_transitions, V).T
        expected_reward = Q_table - gamma * EV

        assert jnp.allclose(reward_table, expected_reward, atol=1e-5)

    def test_adam_optimizer(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Adam optimizer should produce valid results."""
        config = IQLearnConfig(
            optimizer="adam", learning_rate=0.05, max_iter=200, verbose=False
        )
        estimator = IQLearnEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert result.policy is not None
        policy_sum = result.policy.sum(axis=1)
        assert jnp.allclose(policy_sum, jnp.ones(3), atol=1e-5)
