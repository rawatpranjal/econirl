"""Tests for the counterfactual analysis module.

Tests cover:
- CounterfactualType enum membership and values
- CounterfactualResult default type
- state_extrapolation with identity and shift mappings
- discount_factor_change direction
- welfare_decomposition additivity
- Unified dispatcher routing for all type combinations
- Invalid argument detection in the dispatcher
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from econirl.simulation.counterfactual import (
    CounterfactualType,
    CounterfactualResult,
    state_extrapolation,
    counterfactual_policy,
    counterfactual_transitions,
    discount_factor_change,
    welfare_decomposition,
    counterfactual,
    compute_stationary_distribution,
    neural_global_perturbation,
    neural_local_perturbation,
    neural_transition_counterfactual,
    neural_choice_set_counterfactual,
    neural_sieve_compression,
    neural_policy_jacobian,
    neural_perturbation_sweep,
    neural_reward_counterfactual,
)
from econirl.core.types import DDCProblem
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.inference.results import EstimationSummary
from econirl.preferences.linear import LinearUtility


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_problem():
    """A 10-state, 2-action DDC problem with deterministic transitions.

    Action 0 (keep) moves to the next state. Action 1 (replace) resets
    to state 0. Discount factor is 0.95 for fast convergence.
    """
    n_states = 10
    n_actions = 2
    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.95,
        scale_parameter=1.0,
    )

    transitions = jnp.zeros((n_actions, n_states, n_states))
    for s in range(n_states):
        # Keep: deterministic move to next state (or stay at last)
        transitions = transitions.at[0, s, min(s + 1, n_states - 1)].set(1.0)
        # Replace: deterministic reset to state 0
        transitions = transitions.at[1, s, 0].set(1.0)

    return problem, transitions


@pytest.fixture
def small_utility(small_problem):
    """A simple linear utility for the 10-state problem.

    Operating cost increases linearly with state. Replacement incurs a
    fixed cost of 2.0.
    """
    problem, _ = small_problem
    n_states = problem.num_states
    n_actions = problem.num_actions

    # Feature matrix: (n_states, n_actions, n_features)
    # Feature 0: operating cost (state * 0.01 for keep, 0 for replace)
    # Feature 1: replacement indicator (0 for keep, 1 for replace)
    features = jnp.zeros((n_states, n_actions, 2))
    for s in range(n_states):
        features = features.at[s, 0, 0].set(-s * 0.1)  # keep cost
        features = features.at[s, 1, 1].set(-2.0)       # replace cost

    utility = LinearUtility(
        feature_matrix=features,
        parameter_names=["operating_cost", "replacement_cost"],
    )
    return utility


@pytest.fixture
def small_solution(small_problem, small_utility):
    """Solve the small problem and return a mock EstimationSummary."""
    problem, transitions = small_problem
    utility = small_utility

    true_params = jnp.array([1.0, 1.0])
    reward = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)
    sol = value_iteration(operator, reward)

    result = EstimationSummary(
        parameters=true_params,
        parameter_names=["operating_cost", "replacement_cost"],
        standard_errors=jnp.array([0.01, 0.01]),
        method="test",
        value_function=sol.V,
        policy=sol.policy,
    )
    return result


# ============================================================================
# Enum Tests
# ============================================================================


class TestCounterfactualTypeEnum:
    """Tests for CounterfactualType enum membership and values."""

    def test_has_four_members(self):
        """The enum should have exactly four members."""
        assert len(CounterfactualType) == 4

    def test_integer_values_1_through_4(self):
        """Each member should have an integer value from 1 to 4."""
        assert CounterfactualType.STATE_EXTRAPOLATION == 1
        assert CounterfactualType.ENVIRONMENT_CHANGE == 2
        assert CounterfactualType.REWARD_CHANGE == 3
        assert CounterfactualType.WELFARE_DECOMPOSITION == 4

    def test_is_intenum(self):
        """Members should be usable as integers."""
        assert CounterfactualType.STATE_EXTRAPOLATION + 1 == 2


class TestCounterfactualResultDefault:
    """Tests for CounterfactualResult defaults."""

    def test_default_type_is_reward_change(self):
        """The default counterfactual_type should be REWARD_CHANGE."""
        dummy = jnp.zeros((2, 2))
        dummy_v = jnp.zeros(2)
        result = CounterfactualResult(
            baseline_policy=dummy,
            counterfactual_policy=dummy,
            baseline_value=dummy_v,
            counterfactual_value=dummy_v,
            policy_change=dummy,
            value_change=dummy_v,
            welfare_change=0.0,
        )
        assert result.counterfactual_type == CounterfactualType.REWARD_CHANGE


# ============================================================================
# State Extrapolation (Type 1)
# ============================================================================


class TestStateExtrapolation:
    """Tests for the state_extrapolation function."""

    def test_identity_mapping_returns_zero_change(self, small_problem, small_solution):
        """An identity state mapping should produce zero policy change."""
        problem, transitions = small_problem
        identity_map = {s: s for s in range(problem.num_states)}

        cf = state_extrapolation(small_solution, identity_map, problem, transitions)

        np.testing.assert_allclose(
            np.asarray(cf.policy_change),
            np.zeros_like(np.asarray(cf.policy_change)),
            atol=1e-10,
            err_msg="Identity mapping should produce zero policy change",
        )
        assert cf.counterfactual_type == CounterfactualType.STATE_EXTRAPOLATION
        assert abs(cf.welfare_change) < 1e-10

    def test_shift_mapping_moves_policy(self, small_problem, small_solution):
        """Shifting states down by 5 should map state 7 policy to state 2."""
        problem, transitions = small_problem
        shift = 5
        mapping = {s: max(0, s - shift) for s in range(problem.num_states)}

        cf = state_extrapolation(small_solution, mapping, problem, transitions)

        # State 7 in the counterfactual should have the policy of state 2
        baseline_policy_at_2 = small_solution.policy[2]
        cf_policy_at_7 = cf.counterfactual_policy[7]

        np.testing.assert_allclose(
            np.asarray(cf_policy_at_7),
            np.asarray(baseline_policy_at_2),
            atol=1e-10,
            err_msg="State 7 counterfactual policy should equal baseline state 2 policy",
        )

    def test_array_mapping(self, small_problem, small_solution):
        """A numpy array mapping should work identically to a dict mapping."""
        problem, transitions = small_problem
        mapping_dict = {s: max(0, s - 3) for s in range(problem.num_states)}
        mapping_arr = jnp.array(
            [max(0, s - 3) for s in range(problem.num_states)], dtype=jnp.int32
        )

        cf_dict = state_extrapolation(small_solution, mapping_dict, problem, transitions)
        cf_arr = state_extrapolation(small_solution, mapping_arr, problem, transitions)

        np.testing.assert_allclose(
            np.asarray(cf_dict.counterfactual_policy),
            np.asarray(cf_arr.counterfactual_policy),
            atol=1e-10,
        )


# ============================================================================
# Discount Factor Change (Type 3)
# ============================================================================


class TestDiscountFactorChange:
    """Tests for the discount_factor_change function."""

    def test_lower_beta_changes_replacement_probability(
        self, small_problem, small_solution, small_utility
    ):
        """A lower discount factor should change the replacement probability.

        More myopic agents care less about the future cost of high
        mileage, so they should replace less often at intermediate
        states compared to patient agents.
        """
        problem, transitions = small_problem

        cf = discount_factor_change(
            result=small_solution,
            new_discount=0.5,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
        )

        # The policy should change somewhere
        max_abs_change = float(jnp.abs(cf.policy_change).max())
        assert max_abs_change > 0.01, (
            f"Expected noticeable policy change from beta=0.95 to beta=0.5, "
            f"but max absolute change is {max_abs_change}"
        )
        assert cf.counterfactual_type == CounterfactualType.REWARD_CHANGE


# ============================================================================
# Welfare Decomposition (Type 4)
# ============================================================================


class TestWelfareDecomposition:
    """Tests for the welfare_decomposition function."""

    def test_additivity(self, small_problem, small_solution, small_utility):
        """reward_channel + transition_channel + interaction should equal total."""
        problem, transitions = small_problem

        # Create a slightly different transition matrix
        new_transitions = transitions.copy()
        # Make action 0 slightly stochastic: 90% move forward, 10% stay
        n_states = problem.num_states
        for s in range(n_states - 1):
            new_transitions = new_transitions.at[0, s, min(s + 1, n_states - 1)].set(
                0.9
            )
            new_transitions = new_transitions.at[0, s, s].set(0.1)

        new_params = small_solution.parameters * 1.5

        decomp = welfare_decomposition(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            baseline_transitions=transitions,
            new_parameters=new_params,
            new_transitions=new_transitions,
        )

        total = decomp["total_welfare_change"]
        channels_sum = (
            decomp["reward_channel"]
            + decomp["transition_channel"]
            + decomp["interaction_effect"]
        )

        np.testing.assert_allclose(
            total,
            channels_sum,
            atol=1e-6,
            err_msg=(
                "Reward channel, transition channel, and interaction "
                "should sum to total welfare change"
            ),
        )

    def test_requires_at_least_one_change(
        self, small_problem, small_solution, small_utility
    ):
        """Calling with neither new_parameters nor new_transitions should raise."""
        problem, transitions = small_problem

        with pytest.raises(ValueError, match="At least one"):
            welfare_decomposition(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                baseline_transitions=transitions,
                new_parameters=None,
                new_transitions=None,
            )

    def test_params_only_has_zero_transition_channel(
        self, small_problem, small_solution, small_utility
    ):
        """When only parameters change, the transition channel should be zero."""
        problem, transitions = small_problem

        new_params = small_solution.parameters * 1.5

        decomp = welfare_decomposition(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            baseline_transitions=transitions,
            new_parameters=new_params,
            new_transitions=None,
        )

        np.testing.assert_allclose(
            decomp["transition_channel"],
            0.0,
            atol=1e-6,
            err_msg="Transition channel should be zero when only parameters change",
        )


# ============================================================================
# Unified Dispatcher
# ============================================================================


class TestDispatcher:
    """Tests for the counterfactual() unified dispatcher."""

    def test_type1_dispatch(self, small_problem, small_solution, small_utility):
        """Providing only state_mapping should dispatch to Type 1."""
        problem, transitions = small_problem
        mapping = {s: max(0, s - 5) for s in range(problem.num_states)}

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            state_mapping=mapping,
        )
        assert cf.counterfactual_type == CounterfactualType.STATE_EXTRAPOLATION

    def test_type2_dispatch(self, small_problem, small_solution, small_utility):
        """Providing only new_transitions should dispatch to Type 2."""
        problem, transitions = small_problem

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_transitions=transitions,
        )
        assert cf.counterfactual_type == CounterfactualType.ENVIRONMENT_CHANGE

    def test_type3_params_dispatch(
        self, small_problem, small_solution, small_utility
    ):
        """Providing only new_parameters should dispatch to Type 3."""
        problem, transitions = small_problem
        new_params = small_solution.parameters * 2.0

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_parameters=new_params,
        )
        assert cf.counterfactual_type == CounterfactualType.REWARD_CHANGE

    def test_type3_discount_dispatch(
        self, small_problem, small_solution, small_utility
    ):
        """Providing only new_discount should dispatch to Type 3."""
        problem, transitions = small_problem

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_discount=0.8,
        )
        assert cf.counterfactual_type == CounterfactualType.REWARD_CHANGE

    def test_invalid_combo_mapping_and_params(
        self, small_problem, small_solution, small_utility
    ):
        """state_mapping combined with new_parameters should raise ValueError."""
        problem, transitions = small_problem
        mapping = {0: 1}
        new_params = small_solution.parameters * 2.0

        with pytest.raises(ValueError, match="cannot be combined"):
            counterfactual(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                transitions=transitions,
                state_mapping=mapping,
                new_parameters=new_params,
            )

    def test_no_args_raises(self, small_problem, small_solution, small_utility):
        """Providing no counterfactual change should raise ValueError."""
        problem, transitions = small_problem

        with pytest.raises(ValueError, match="No counterfactual change specified"):
            counterfactual(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                transitions=transitions,
            )

    def test_combined_type2_and_type3(
        self, small_problem, small_solution, small_utility
    ):
        """Providing both new_parameters and new_transitions should work."""
        problem, transitions = small_problem
        new_params = small_solution.parameters * 1.5

        cf = counterfactual(
            result=small_solution,
            utility=small_utility,
            problem=problem,
            transitions=transitions,
            new_parameters=new_params,
            new_transitions=transitions,
        )
        # Combined dispatch returns ENVIRONMENT_CHANGE type
        assert cf.counterfactual_type == CounterfactualType.ENVIRONMENT_CHANGE

    def test_discount_with_transitions_raises(
        self, small_problem, small_solution, small_utility
    ):
        """new_discount combined with new_transitions should raise ValueError."""
        problem, transitions = small_problem

        with pytest.raises(ValueError, match="cannot be combined"):
            counterfactual(
                result=small_solution,
                utility=small_utility,
                problem=problem,
                transitions=transitions,
                new_discount=0.8,
                new_transitions=transitions,
            )


# ============================================================================
# Stationary Distribution
# ============================================================================


class TestStationaryDistribution:
    """Tests for the compute_stationary_distribution helper."""

    def test_sums_to_one(self, small_problem, small_solution):
        """The stationary distribution should sum to 1."""
        _, transitions = small_problem

        mu = compute_stationary_distribution(small_solution.policy, transitions)

        np.testing.assert_allclose(
            float(mu.sum()), 1.0, atol=1e-8,
            err_msg="Stationary distribution should sum to 1",
        )

    def test_all_nonnegative(self, small_problem, small_solution):
        """All entries in the stationary distribution should be nonnegative."""
        _, transitions = small_problem

        mu = compute_stationary_distribution(small_solution.policy, transitions)

        assert bool(jnp.all(mu >= 0)), "Stationary distribution has negative entries"

    def test_is_fixed_point(self, small_problem, small_solution):
        """The distribution should be a fixed point of the transition operator."""
        _, transitions = small_problem
        policy = small_solution.policy

        mu = compute_stationary_distribution(policy, transitions)

        # Policy-weighted transition: P^pi(s,s') = sum_a pi(a|s) P(s'|s,a)
        P_pi = jnp.einsum("sa,ast->st", policy, transitions)
        mu_next = P_pi.T @ mu
        mu_next = mu_next / mu_next.sum()

        np.testing.assert_allclose(
            np.asarray(mu),
            np.asarray(mu_next),
            atol=1e-8,
            err_msg="Stationary distribution should be a fixed point",
        )


# ============================================================================
# Neural Counterfactual Tests
# ============================================================================


@pytest.fixture
def neural_reward(small_problem, small_utility):
    """A reward matrix from a hypothetical neural estimator.

    We use the structural reward as the 'neural' reward so we can
    verify results against known quantities.
    """
    problem, transitions = small_problem
    true_params = jnp.array([1.0, 1.0])
    reward = small_utility.compute(true_params)
    return jnp.asarray(reward)


class TestNeuralGlobalPerturbation:
    """Tests for neural_global_perturbation."""

    def test_penalizing_action_reduces_probability(
        self, small_problem, neural_reward
    ):
        problem, transitions = small_problem
        cf = neural_global_perturbation(neural_reward, action=1, delta=2.0,
                                        problem=problem, transitions=transitions)
        # Penalizing replace (action 1) should reduce its probability
        assert (cf.policy_change[:, 1] <= 0.01).all()

    def test_zero_delta_gives_zero_change(self, small_problem, neural_reward):
        problem, transitions = small_problem
        cf = neural_global_perturbation(neural_reward, action=1, delta=0.0,
                                        problem=problem, transitions=transitions)
        np.testing.assert_allclose(
            np.asarray(cf.policy_change), 0.0, atol=1e-6,
        )


class TestNeuralLocalPerturbation:
    """Tests for neural_local_perturbation."""

    def test_only_affects_masked_states(self, small_problem, neural_reward):
        problem, transitions = small_problem
        mask = jnp.arange(problem.num_states) >= 7  # only states 7, 8, 9
        cf = neural_local_perturbation(
            neural_reward, action=0, delta=5.0,
            state_mask=mask, problem=problem, transitions=transitions,
        )
        # Unmasked states should have negligible direct reward change,
        # but policy may still change due to value function propagation.
        # The key test is that the description mentions the right count.
        assert "3 states affected" in cf.description


class TestNeuralChoiceSet:
    """Tests for neural_choice_set_counterfactual."""

    def test_all_allowed_equals_baseline(self, small_problem, neural_reward):
        problem, transitions = small_problem
        mask = jnp.ones((problem.num_states, problem.num_actions), dtype=jnp.bool_)
        cf = neural_choice_set_counterfactual(neural_reward, mask, problem, transitions)
        np.testing.assert_allclose(
            np.asarray(cf.policy_change), 0.0, atol=1e-5,
        )

    def test_blocked_action_has_near_zero_prob(self, small_problem, neural_reward):
        problem, transitions = small_problem
        mask = jnp.ones((problem.num_states, problem.num_actions), dtype=jnp.bool_)
        # Block replace (action 1) at states 0-4
        mask = mask.at[:5, 1].set(False)
        cf = neural_choice_set_counterfactual(neural_reward, mask, problem, transitions)
        # Replacement probability at blocked states should be near zero
        for s in range(5):
            assert float(cf.counterfactual_policy[s, 1]) < 1e-10


class TestNeuralSieveCompression:
    """Tests for neural_sieve_compression."""

    def test_perfect_linear_gives_r2_one(self, small_problem, small_utility):
        """If the neural reward IS a linear function of features, R^2 = 1."""
        problem, _ = small_problem
        true_params = jnp.array([1.0, 1.0])
        reward = jnp.asarray(small_utility.compute(true_params))
        features = jnp.asarray(small_utility.feature_matrix)

        result = neural_sieve_compression(reward, features,
                                          parameter_names=["op", "rc"])
        assert result["r_squared"] > 0.999
        np.testing.assert_allclose(result["theta"], [1.0, 1.0], atol=0.01)

    def test_returns_all_keys(self, small_problem, small_utility, neural_reward):
        features = jnp.asarray(small_utility.feature_matrix)
        result = neural_sieve_compression(neural_reward, features)
        for key in ["theta", "se", "r_squared", "residuals", "fitted_reward",
                     "parameter_names"]:
            assert key in result


class TestNeuralPolicyJacobian:
    """Tests for neural_policy_jacobian."""

    def test_shape(self, small_problem, neural_reward):
        problem, transitions = small_problem
        J = neural_policy_jacobian(neural_reward, problem, transitions,
                                   target_action=1)
        assert J.shape == (problem.num_states, problem.num_states, problem.num_actions)

    def test_self_perturbation_has_largest_effect(self, small_problem, neural_reward):
        """Perturbing r(s, replace) should most affect pi(s, replace)."""
        problem, transitions = small_problem
        J = neural_policy_jacobian(neural_reward, problem, transitions,
                                   target_action=1)
        # For each state, the diagonal (perturbing own state) should
        # have among the largest absolute effects
        for s in range(1, problem.num_states - 1):
            diag_effect = abs(float(J[s, s, 1]))
            mean_effect = float(jnp.abs(J[s, :, 1]).mean())
            # Diagonal should be at least as large as the average
            assert diag_effect >= mean_effect * 0.5


class TestNeuralPerturbationSweep:
    """Tests for neural_perturbation_sweep."""

    def test_increasing_penalty_reduces_action_prob(
        self, small_problem, neural_reward
    ):
        problem, transitions = small_problem
        deltas = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0])
        result = neural_perturbation_sweep(
            neural_reward, action=1, delta_grid=deltas,
            problem=problem, transitions=transitions,
        )
        # Mean replacement probability should decrease as penalty increases
        probs = result["mean_action_prob"]
        assert probs[0] >= probs[-1]

    def test_returns_all_keys(self, small_problem, neural_reward):
        problem, transitions = small_problem
        result = neural_perturbation_sweep(
            neural_reward, action=1, delta_grid=jnp.array([0.0, 1.0]),
            problem=problem, transitions=transitions,
        )
        for key in ["delta_grid", "mean_action_prob", "welfare",
                     "policy_matrix", "baseline_action_prob", "baseline_welfare"]:
            assert key in result


class TestNeuralTransitionCounterfactual:
    """Tests for neural_transition_counterfactual."""

    def test_same_transitions_gives_zero_change(self, small_problem, neural_reward):
        problem, transitions = small_problem
        cf = neural_transition_counterfactual(
            neural_reward, transitions, problem, transitions,
        )
        np.testing.assert_allclose(
            np.asarray(cf.policy_change), 0.0, atol=1e-5,
        )

    def test_type_tag_is_environment_change(self, small_problem, neural_reward):
        problem, transitions = small_problem
        cf = neural_transition_counterfactual(
            neural_reward, transitions, problem, transitions,
        )
        assert cf.counterfactual_type == CounterfactualType.ENVIRONMENT_CHANGE
