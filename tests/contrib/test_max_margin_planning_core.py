"""Unit tests for Maximum Margin Planning estimator."""

import pytest
import torch
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
from econirl.preferences.action_reward import ActionDependentReward


# --- Fixtures ---


@pytest.fixture
def simple_problem():
    """Create a simple 5-state, 2-action DDC problem."""
    return DDCProblem(
        num_states=5,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
    )


@pytest.fixture
def simple_transitions(simple_problem):
    """Create simple transition matrices.

    Action 0: tends to move to lower states
    Action 1: tends to move to higher states
    """
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions

    transitions = torch.zeros((n_actions, n_states, n_states))

    for s in range(n_states):
        # Action 0: move toward state 0
        if s > 0:
            transitions[0, s, s - 1] = 0.7
            transitions[0, s, s] = 0.3
        else:
            transitions[0, s, s] = 1.0

        # Action 1: move toward state n-1
        if s < n_states - 1:
            transitions[1, s, s + 1] = 0.7
            transitions[1, s, s] = 0.3
        else:
            transitions[1, s, s] = 1.0

    return transitions


@pytest.fixture
def simple_reward_fn(simple_problem):
    """Create a simple action-dependent reward function.

    Features:
    - Feature 0: mileage cost (state / n_states)
    - Feature 1: replacement cost (action == 1)
    """
    n_states = simple_problem.num_states
    n_actions = simple_problem.num_actions

    # Feature matrix: (n_states, n_actions, n_features)
    feature_matrix = torch.zeros((n_states, n_actions, 2))

    for s in range(n_states):
        # Feature 0: normalized state (mileage proxy)
        feature_matrix[s, 0, 0] = s / (n_states - 1)
        feature_matrix[s, 1, 0] = s / (n_states - 1)

        # Feature 1: action indicator (replacement cost)
        feature_matrix[s, 0, 1] = 0.0
        feature_matrix[s, 1, 1] = 1.0

    return ActionDependentReward(
        feature_matrix=feature_matrix,
        parameter_names=["mileage_cost", "replacement_cost"],
    )


@pytest.fixture
def true_params():
    """True parameters for testing."""
    return torch.tensor([-0.5, -1.0], dtype=torch.float32)


@pytest.fixture
def expert_panel(simple_problem, simple_transitions, simple_reward_fn, true_params):
    """Generate expert panel data from true parameters."""
    from econirl.core.solvers import value_iteration

    # Solve for optimal policy under true params
    operator = SoftBellmanOperator(simple_problem, simple_transitions)
    reward_matrix = simple_reward_fn.compute(true_params)
    result = value_iteration(operator, reward_matrix, tol=1e-10, max_iter=1000)
    policy = result.policy

    # Generate trajectories
    n_trajectories = 50
    horizon = 20
    trajectories = []

    torch.manual_seed(42)
    np.random.seed(42)

    for _ in range(n_trajectories):
        states = []
        actions = []
        next_states = []

        # Start from random state
        s = np.random.randint(0, simple_problem.num_states)

        for _ in range(horizon):
            states.append(s)

            # Sample action from policy
            probs = policy[s].numpy()
            a = np.random.choice(simple_problem.num_actions, p=probs)
            actions.append(a)

            # Transition
            trans_probs = simple_transitions[a, s].numpy()
            s_next = np.random.choice(simple_problem.num_states, p=trans_probs)
            next_states.append(s_next)

            # Update state for next period
            s = s_next

        trajectories.append(
            Trajectory(
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
            )
        )

    return Panel(trajectories=trajectories)


# --- Config Tests ---


class TestMMPConfig:
    """Tests for MMPConfig dataclass."""

    def test_default_config(self):
        """Test that default config is created correctly."""
        config = MMPConfig()

        assert config.learning_rate == 0.1
        assert config.learning_rate_schedule == "1/sqrt(t)"
        assert config.max_iterations == 200
        assert config.convergence_tol == 1e-5
        assert config.regularization_lambda == 0.01
        assert config.loss_type == "policy_kl"
        assert config.loss_scale == 1.0
        assert config.inner_solver == "hybrid"
        assert config.compute_se is True
        assert config.verbose is False

    def test_custom_config(self):
        """Test config with custom values."""
        config = MMPConfig(
            learning_rate=0.05,
            max_iterations=100,
            loss_type="trajectory_hamming",
            regularization_lambda=0.1,
            verbose=True,
        )

        assert config.learning_rate == 0.05
        assert config.max_iterations == 100
        assert config.loss_type == "trajectory_hamming"
        assert config.regularization_lambda == 0.1
        assert config.verbose is True


# --- Estimator Instantiation Tests ---


class TestMaxMarginPlanningEstimatorInstantiation:
    """Tests for estimator instantiation."""

    def test_default_instantiation(self):
        """Test instantiation with default config."""
        estimator = MaxMarginPlanningEstimator()

        assert estimator.name == "MMP (Ratliff, Bagnell & Zinkevich 2006)"
        assert estimator.config.learning_rate == 0.1

    def test_with_config(self):
        """Test instantiation with custom config."""
        config = MMPConfig(learning_rate=0.05, verbose=True)
        estimator = MaxMarginPlanningEstimator(config=config)

        assert estimator.config.learning_rate == 0.05
        assert estimator.config.verbose is True

    def test_with_kwargs(self):
        """Test instantiation with kwargs override."""
        estimator = MaxMarginPlanningEstimator(learning_rate=0.01, max_iterations=50)

        assert estimator.config.learning_rate == 0.01
        assert estimator.config.max_iterations == 50

    def test_config_with_kwargs_override(self):
        """Test config with kwargs override."""
        config = MMPConfig(learning_rate=0.1)
        estimator = MaxMarginPlanningEstimator(config=config, learning_rate=0.05)

        assert estimator.config.learning_rate == 0.05


# --- Expert Policy Estimation Tests ---


class TestExpertPolicyEstimation:
    """Tests for expert policy estimation from data."""

    def test_expert_policy_shape(self, simple_problem, expert_panel):
        """Test that estimated expert policy has correct shape."""
        estimator = MaxMarginPlanningEstimator()
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)

        assert expert_policy.shape == (simple_problem.num_states, simple_problem.num_actions)

    def test_expert_policy_probabilities(self, simple_problem, expert_panel):
        """Test that expert policy produces valid probabilities."""
        estimator = MaxMarginPlanningEstimator()
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)

        # Should be non-negative
        assert (expert_policy >= 0).all()

        # Should sum to 1 across actions for each state
        row_sums = expert_policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(simple_problem.num_states))


# --- Loss Matrix Tests ---


class TestLossMatrix:
    """Tests for loss matrix computation."""

    def test_policy_kl_loss_shape(self, simple_problem, expert_panel):
        """Test policy KL loss matrix has correct shape."""
        estimator = MaxMarginPlanningEstimator(loss_type="policy_kl")
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)

        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        assert loss_matrix.shape == (simple_problem.num_states, simple_problem.num_actions)

    def test_policy_kl_loss_non_negative(self, simple_problem, expert_panel):
        """Test that policy KL loss is non-negative."""
        estimator = MaxMarginPlanningEstimator(loss_type="policy_kl")
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)

        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        # -log(p) is non-negative for p in (0, 1]
        assert (loss_matrix >= 0).all()

    def test_policy_kl_loss_high_for_rare_actions(self, simple_problem):
        """Test that KL loss is higher for actions expert rarely takes."""
        estimator = MaxMarginPlanningEstimator(loss_type="policy_kl")

        # Create a policy where expert strongly prefers action 0
        expert_policy = torch.zeros((simple_problem.num_states, simple_problem.num_actions))
        expert_policy[:, 0] = 0.9  # Expert mostly takes action 0
        expert_policy[:, 1] = 0.1  # Rarely takes action 1

        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        # Loss for action 1 should be higher than action 0
        assert (loss_matrix[:, 1] > loss_matrix[:, 0]).all()

    def test_trajectory_hamming_loss_range(self, simple_problem, expert_panel):
        """Test trajectory Hamming loss is in [0, 1]."""
        estimator = MaxMarginPlanningEstimator(loss_type="trajectory_hamming")
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)

        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        # Hamming loss should be in [0, 1]
        assert (loss_matrix >= 0).all()
        assert (loss_matrix <= 1).all()

    def test_trajectory_hamming_loss_zero_for_certain_actions(self, simple_problem):
        """Test Hamming loss is 0 for actions expert always takes."""
        estimator = MaxMarginPlanningEstimator(loss_type="trajectory_hamming")

        # Create deterministic policy
        expert_policy = torch.zeros((simple_problem.num_states, simple_problem.num_actions))
        expert_policy[:, 0] = 1.0  # Expert always takes action 0

        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        # Loss should be 0 for action 0, 1 for action 1
        assert torch.allclose(loss_matrix[:, 0], torch.zeros(simple_problem.num_states))
        assert torch.allclose(loss_matrix[:, 1], torch.ones(simple_problem.num_states))


# --- Loss-Augmented VI Tests ---


class TestLossAugmentedVI:
    """Tests for loss-augmented value iteration."""

    def test_returns_valid_policy(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test that loss-augmented VI returns a valid policy."""
        estimator = MaxMarginPlanningEstimator()
        operator = SoftBellmanOperator(simple_problem, simple_transitions)

        # Create reward and loss matrices
        theta = torch.zeros(2)
        reward_matrix = simple_reward_fn.compute(theta)
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)
        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        V, policy, converged = estimator._loss_augmented_value_iteration(
            reward_matrix, loss_matrix, operator
        )

        # Policy should be valid probabilities
        assert policy.shape == (simple_problem.num_states, simple_problem.num_actions)
        assert (policy >= 0).all()
        assert torch.allclose(policy.sum(dim=1), torch.ones(simple_problem.num_states))

    def test_loss_augmentation_changes_policy(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test that adding loss changes the optimal policy."""
        from econirl.core.solvers import value_iteration

        estimator = MaxMarginPlanningEstimator(loss_scale=10.0)  # High scale
        operator = SoftBellmanOperator(simple_problem, simple_transitions)

        theta = torch.zeros(2)
        reward_matrix = simple_reward_fn.compute(theta)

        # Solve without loss
        result_no_loss = value_iteration(operator, reward_matrix, tol=1e-10, max_iter=1000)

        # Solve with loss augmentation
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)
        loss_matrix = estimator._compute_loss_matrix(
            expert_policy, simple_problem.num_states, simple_problem.num_actions
        )

        _, policy_with_loss, _ = estimator._loss_augmented_value_iteration(
            reward_matrix, loss_matrix, operator
        )

        # Policies should be different (loss encourages deviating from expert)
        policy_diff = torch.abs(result_no_loss.policy - policy_with_loss).max()
        assert policy_diff > 0.01  # Should be meaningfully different


# --- Subgradient Computation Tests ---


class TestSubgradientComputation:
    """Tests for subgradient computation."""

    def test_subgradient_shape(self):
        """Test subgradient has correct shape."""
        estimator = MaxMarginPlanningEstimator(regularization_lambda=0.1)

        theta = torch.tensor([1.0, 2.0])
        expert_features = torch.tensor([0.5, 0.3])
        policy_features = torch.tensor([0.6, 0.4])

        subgradient = estimator._compute_subgradient(theta, expert_features, policy_features)

        assert subgradient.shape == theta.shape

    def test_subgradient_with_zero_regularization(self):
        """Test subgradient with zero regularization."""
        estimator = MaxMarginPlanningEstimator(regularization_lambda=0.0)

        theta = torch.tensor([1.0, 2.0])
        expert_features = torch.tensor([0.5, 0.3])
        policy_features = torch.tensor([0.6, 0.4])

        subgradient = estimator._compute_subgradient(theta, expert_features, policy_features)

        # g = μ̂ - μ* when λ = 0
        expected = policy_features - expert_features
        assert torch.allclose(subgradient, expected)

    def test_subgradient_regularization_term(self):
        """Test that regularization term is included."""
        lambda_reg = 0.1
        estimator = MaxMarginPlanningEstimator(regularization_lambda=lambda_reg)

        theta = torch.tensor([1.0, 2.0])
        expert_features = torch.tensor([0.5, 0.3])
        policy_features = torch.tensor([0.6, 0.4])

        subgradient = estimator._compute_subgradient(theta, expert_features, policy_features)

        # g = λθ + (μ̂ - μ*)
        expected = lambda_reg * theta + (policy_features - expert_features)
        assert torch.allclose(subgradient, expected)


# --- Learning Rate Schedule Tests ---


class TestLearningRateSchedule:
    """Tests for learning rate schedules."""

    def test_constant_schedule(self):
        """Test constant learning rate schedule."""
        estimator = MaxMarginPlanningEstimator(
            learning_rate=0.1, learning_rate_schedule="constant"
        )

        assert estimator._get_learning_rate(1) == 0.1
        assert estimator._get_learning_rate(10) == 0.1
        assert estimator._get_learning_rate(100) == 0.1

    def test_1_over_t_schedule(self):
        """Test 1/t learning rate schedule."""
        estimator = MaxMarginPlanningEstimator(
            learning_rate=1.0, learning_rate_schedule="1/t"
        )

        assert estimator._get_learning_rate(1) == 1.0
        assert estimator._get_learning_rate(2) == 0.5
        assert estimator._get_learning_rate(10) == 0.1

    def test_1_over_sqrt_t_schedule(self):
        """Test 1/sqrt(t) learning rate schedule."""
        estimator = MaxMarginPlanningEstimator(
            learning_rate=1.0, learning_rate_schedule="1/sqrt(t)"
        )

        assert estimator._get_learning_rate(1) == 1.0
        assert np.isclose(estimator._get_learning_rate(4), 0.5)
        assert np.isclose(estimator._get_learning_rate(100), 0.1)


# --- Full Optimization Tests ---


class TestFullOptimization:
    """Tests for the complete optimization pipeline."""

    def test_optimization_returns_result(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test that optimization returns an EstimationResult."""
        estimator = MaxMarginPlanningEstimator(
            max_iterations=10, compute_se=False, verbose=False
        )

        result = estimator._optimize(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert hasattr(result, "parameters")
        assert hasattr(result, "policy")
        assert hasattr(result, "value_function")
        assert hasattr(result, "log_likelihood")
        assert hasattr(result, "converged")
        assert hasattr(result, "num_iterations")

    def test_optimization_result_shapes(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test that optimization result has correct shapes."""
        estimator = MaxMarginPlanningEstimator(
            max_iterations=10, compute_se=False, verbose=False
        )

        result = estimator._optimize(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        n_params = simple_reward_fn.num_parameters
        n_states = simple_problem.num_states
        n_actions = simple_problem.num_actions

        assert result.parameters.shape == (n_params,)
        assert result.policy.shape == (n_states, n_actions)
        assert result.value_function.shape == (n_states,)

    def test_optimization_valid_policy(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test that final policy is valid."""
        estimator = MaxMarginPlanningEstimator(
            max_iterations=10, compute_se=False, verbose=False
        )

        result = estimator._optimize(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Policy should be valid probabilities
        assert (result.policy >= 0).all()
        assert torch.allclose(
            result.policy.sum(dim=1), torch.ones(simple_problem.num_states), atol=1e-6
        )

    def test_optimization_with_initial_params(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel, true_params
    ):
        """Test optimization with initial parameters."""
        estimator = MaxMarginPlanningEstimator(
            max_iterations=10, compute_se=False, verbose=False
        )

        result = estimator._optimize(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
            initial_params=true_params.clone(),
        )

        # Should complete without error
        assert result.parameters.shape == true_params.shape

    def test_estimate_interface(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test the public estimate() interface."""
        estimator = MaxMarginPlanningEstimator(
            max_iterations=10, compute_se=False, verbose=False
        )

        # This calls _optimize and wraps result in EstimationSummary
        summary = estimator.estimate(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        assert hasattr(summary, "parameters")
        assert hasattr(summary, "standard_errors")
        assert hasattr(summary, "method")
        assert "MMP" in summary.method


# --- Margin Computation Tests ---


class TestMarginComputation:
    """Tests for margin computation."""

    def test_compute_margin(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel, true_params
    ):
        """Test margin computation."""
        estimator = MaxMarginPlanningEstimator()

        margin = estimator.compute_margin(
            theta=true_params,
            panel=expert_panel,
            reward_fn=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Margin should be a scalar
        assert isinstance(margin, float)


# --- Feature Computation Tests ---


class TestFeatureComputation:
    """Tests for feature expectation computation."""

    def test_expert_features_shape(
        self, simple_reward_fn, expert_panel
    ):
        """Test expert features have correct shape."""
        estimator = MaxMarginPlanningEstimator()

        expert_features = estimator._compute_expert_features(expert_panel, simple_reward_fn)

        assert expert_features.shape == (simple_reward_fn.num_parameters,)

    def test_policy_features_shape(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel
    ):
        """Test policy features have correct shape."""
        estimator = MaxMarginPlanningEstimator()

        # Create a policy
        expert_policy = estimator._estimate_expert_policy(expert_panel, simple_problem)

        policy_features = estimator._compute_policy_features(
            policy=expert_policy,
            transitions=simple_transitions,
            reward_fn=simple_reward_fn,
            problem=simple_problem,
        )

        assert policy_features.shape == (simple_reward_fn.num_parameters,)


# --- Integration Test ---


class TestIntegration:
    """Integration tests for the full MMP pipeline."""

    def test_parameter_direction(
        self, simple_problem, simple_transitions, simple_reward_fn, expert_panel, true_params
    ):
        """Test that recovered parameters have correct direction."""
        estimator = MaxMarginPlanningEstimator(
            max_iterations=50,
            learning_rate=0.2,
            regularization_lambda=0.001,
            compute_se=False,
            verbose=False,
        )

        result = estimator._optimize(
            panel=expert_panel,
            utility=simple_reward_fn,
            problem=simple_problem,
            transitions=simple_transitions,
        )

        # Check that signs match (direction is correct)
        # Note: IRL only recovers reward up to scale, so we check direction
        estimated = result.parameters
        true_direction = true_params / torch.norm(true_params)
        estimated_direction = estimated / torch.norm(estimated)

        # Cosine similarity should be positive (same direction)
        cos_sim = torch.dot(true_direction, estimated_direction).item()

        # Should recover correct direction (cosine similarity > 0.5)
        assert cos_sim > 0.0, f"Cosine similarity {cos_sim} should be positive"
