"""Tests for GAIL estimator."""

import pytest
import torch

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.gail import GAILEstimator, GAILConfig
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
    # Shape: (n_actions, n_states, n_states) = [a, from_s, to_s]
    transitions = torch.zeros(n_actions, n_states, n_states)

    # Action 0: stay in same state
    for s in range(n_states):
        transitions[0, s, s] = 1.0

    # Action 1: move to next state (cyclic)
    for s in range(n_states):
        next_s = (s + 1) % n_states
        transitions[1, s, next_s] = 1.0

    return transitions


@pytest.fixture
def simple_reward_fn(simple_problem):
    """Create simple action-dependent reward function."""
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions
    # Features: one-hot for action (action 0 is "good", action 1 is "bad")
    features = torch.zeros(n_states, n_actions, 2)
    features[:, 0, 0] = 1.0  # Action 0 feature
    features[:, 1, 1] = 1.0  # Action 1 feature
    return ActionDependentReward(
        feature_matrix=features,
        parameter_names=["action_0_reward", "action_1_reward"],
    )


@pytest.fixture
def expert_panel():
    """Create expert demonstrations favoring action 0."""
    trajectories = []
    for i in range(20):
        # Expert mostly takes action 0 (staying)
        states = torch.tensor([0, 0, 0, 0, 0])
        actions = torch.tensor([0, 0, 0, 0, 0])
        next_states = torch.tensor([0, 0, 0, 0, 0])
        trajectories.append(Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=i,
        ))
    return Panel(trajectories=trajectories)


class TestGAILEstimator:
    """Tests for GAIL estimator."""

    def test_gail_init(self):
        """GAIL should initialize with default config."""
        estimator = GAILEstimator()
        assert estimator.name == "GAIL (Ho & Ermon 2016)"

    def test_gail_init_with_config(self):
        """GAIL should accept custom config."""
        config = GAILConfig(max_rounds=50, discriminator_lr=0.05)
        estimator = GAILEstimator(config=config)
        assert estimator.config.max_rounds == 50
        assert estimator.config.discriminator_lr == 0.05

    def test_gail_estimate_returns_result(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """GAIL estimate should return EstimationSummary."""
        config = GAILConfig(max_rounds=10, verbose=False)
        estimator = GAILEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert result.parameters is not None
        assert result.policy is not None
        assert result.policy.shape == (3, 2)  # (n_states, n_actions)

    def test_gail_policy_is_valid_distribution(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """GAIL policy should sum to 1 for each state."""
        config = GAILConfig(max_rounds=10, verbose=False)
        estimator = GAILEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Policy should sum to 1 for each state
        policy_sum = result.policy.sum(dim=1)
        assert torch.allclose(policy_sum, torch.ones(3), atol=1e-5)

    def test_gail_learns_expert_preference(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """GAIL should learn to prefer action 0 like expert."""
        config = GAILConfig(max_rounds=50, verbose=False)
        estimator = GAILEstimator(config=config)

        result = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Learned policy should prefer action 0 (like expert)
        # In state 0, P(a=0|s=0) should be > P(a=1|s=0)
        assert result.policy[0, 0] > result.policy[0, 1]
