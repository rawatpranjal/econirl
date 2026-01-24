"""Tests for the new sklearn-style utility classes.

These tests cover the Utility base class, LinearCost built-in utility,
CallableUtility wrapper, and make_utility factory function.
"""

import pytest
import numpy as np

from econirl.utilities import (
    Utility,
    LinearCost,
    CallableUtility,
    make_utility,
)


class TestLinearCostNParams:
    """Test LinearCost.n_params property."""

    def test_linear_cost_n_params(self):
        """LinearCost has 2 parameters (theta_c and RC)."""
        utility = LinearCost()
        assert utility.n_params == 2


class TestLinearCostParamNames:
    """Test LinearCost.param_names property."""

    def test_linear_cost_param_names(self):
        """LinearCost has parameter names ['theta_c', 'RC']."""
        utility = LinearCost()
        assert utility.param_names == ["theta_c", "RC"]


class TestLinearCostCall:
    """Test LinearCost.__call__ method."""

    def test_linear_cost_call_no_replace(self):
        """LinearCost computes u = -theta_c * s * (1-a) - RC * a.

        For a=0 (don't replace): u = -theta_c * s
        """
        utility = LinearCost()
        params = np.array([0.001, 3.0])  # theta_c=0.001, RC=3.0

        # State 10, action 0 (don't replace)
        result = utility(state=10, action=0, params=params)
        expected = -0.001 * 10 * 1 - 3.0 * 0  # = -0.01
        assert np.isclose(result, expected)

    def test_linear_cost_call_replace(self):
        """LinearCost computes u = -theta_c * s * (1-a) - RC * a.

        For a=1 (replace): u = -RC
        """
        utility = LinearCost()
        params = np.array([0.001, 3.0])

        # State 10, action 1 (replace)
        result = utility(state=10, action=1, params=params)
        expected = -0.001 * 10 * 0 - 3.0 * 1  # = -3.0
        assert np.isclose(result, expected)

    def test_linear_cost_call_zero_state(self):
        """At state 0, no operating cost."""
        utility = LinearCost()
        params = np.array([0.001, 3.0])

        result = utility(state=0, action=0, params=params)
        assert np.isclose(result, 0.0)

    def test_linear_cost_call_vectorized(self):
        """LinearCost works with array inputs."""
        utility = LinearCost()
        params = np.array([0.001, 3.0])

        states = np.array([0, 10, 20])
        actions = np.array([0, 0, 1])

        result = utility(state=states, action=actions, params=params)
        expected = np.array([0.0, -0.01, -3.0])

        np.testing.assert_allclose(result, expected)


class TestLinearCostMatrix:
    """Test LinearCost.matrix method."""

    def test_linear_cost_matrix_shape(self):
        """LinearCost.matrix returns shape (n_states, 2)."""
        utility = LinearCost()
        params = np.array([0.001, 3.0])

        result = utility.matrix(n_states=90, params=params)

        assert result.shape == (90, 2)

    def test_linear_cost_matrix_values(self):
        """LinearCost.matrix contains correct utility values."""
        utility = LinearCost()
        params = np.array([0.001, 3.0])

        result = utility.matrix(n_states=5, params=params)

        # Check specific values
        # State 0, action 0: -0.001 * 0 = 0
        assert np.isclose(result[0, 0], 0.0)
        # State 0, action 1: -3.0
        assert np.isclose(result[0, 1], -3.0)
        # State 4, action 0: -0.001 * 4 = -0.004
        assert np.isclose(result[4, 0], -0.004)
        # State 4, action 1: -3.0
        assert np.isclose(result[4, 1], -3.0)


class TestCallableUtility:
    """Test CallableUtility wrapper class."""

    def test_callable_utility_basic(self):
        """CallableUtility wraps a custom function."""
        def my_utility(state, action, params):
            return -params[0] * state - params[1] * action

        utility = CallableUtility(
            fn=my_utility,
            n_params=2,
            param_names=["cost", "action_cost"],
        )

        assert utility.n_params == 2
        assert utility.param_names == ["cost", "action_cost"]

    def test_callable_utility_call(self):
        """CallableUtility correctly calls the wrapped function."""
        def my_utility(state, action, params):
            return -params[0] * state - params[1] * action

        utility = CallableUtility(
            fn=my_utility,
            n_params=2,
            param_names=["cost", "action_cost"],
        )

        params = np.array([0.1, 2.0])
        result = utility(state=5, action=1, params=params)

        expected = -0.1 * 5 - 2.0 * 1  # = -2.5
        assert np.isclose(result, expected)

    def test_callable_utility_matrix(self):
        """CallableUtility.matrix works correctly."""
        def my_utility(state, action, params):
            return -params[0] * state

        utility = CallableUtility(
            fn=my_utility,
            n_params=1,
            param_names=["cost"],
        )

        params = np.array([0.01])
        result = utility.matrix(n_states=3, params=params)

        # Shape should be (3, 2) for binary actions
        assert result.shape == (3, 2)
        # All columns should be the same since utility doesn't depend on action
        np.testing.assert_allclose(result[:, 0], result[:, 1])


class TestMakeUtility:
    """Test make_utility factory function."""

    def test_make_utility_creates_callable_utility(self):
        """make_utility returns a CallableUtility instance."""
        def my_fn(state, action, params):
            return 0.0

        utility = make_utility(fn=my_fn, n_params=1, param_names=["p"])

        assert isinstance(utility, CallableUtility)
        assert utility.n_params == 1
        assert utility.param_names == ["p"]

    def test_make_utility_default_param_names(self):
        """make_utility generates default param names if not provided."""
        def my_fn(state, action, params):
            return 0.0

        utility = make_utility(fn=my_fn, n_params=3)

        assert utility.n_params == 3
        assert utility.param_names == ["theta_0", "theta_1", "theta_2"]


class TestUtilityBaseClass:
    """Test the Utility abstract base class."""

    def test_utility_is_abstract(self):
        """Utility cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Utility()

    def test_utility_subclass_must_implement_call(self):
        """Subclasses must implement __call__."""
        class IncompleteUtility(Utility):
            @property
            def n_params(self):
                return 1

            @property
            def param_names(self):
                return ["p"]

        with pytest.raises(TypeError):
            IncompleteUtility()


class TestUtilityParamBounds:
    """Test parameter bounds functionality."""

    def test_linear_cost_param_bounds_default(self):
        """LinearCost has default bounds of (-inf, inf)."""
        utility = LinearCost()

        lower, upper = utility.param_bounds

        assert len(lower) == 2
        assert len(upper) == 2
        assert all(l == float("-inf") for l in lower)
        assert all(u == float("inf") for u in upper)

    def test_linear_cost_param_init_default(self):
        """LinearCost has default initial values."""
        utility = LinearCost()

        init = utility.param_init

        assert len(init) == 2
        assert all(np.isfinite(init))
