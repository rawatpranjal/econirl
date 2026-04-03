"""Tests for multiple cost function specifications in RustBusEnvironment.

Verifies that the cost_type parameter correctly produces different feature
matrices and utility computations, while maintaining backward compatibility
with the default linear cost type.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment, VALID_COST_TYPES


class TestCostTypeBackwardCompat:
    """Default cost_type='linear' must match the original behavior."""

    def test_default_is_linear(self):
        env = RustBusEnvironment()
        assert env.cost_type == "linear"

    def test_linear_feature_shape(self):
        env = RustBusEnvironment(num_mileage_bins=20)
        assert env.feature_matrix.shape == (20, 2, 2)

    def test_linear_parameter_names(self):
        env = RustBusEnvironment()
        assert env.parameter_names == ["operating_cost", "replacement_cost"]

    def test_linear_true_parameters(self):
        env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)
        assert env.true_parameters == {
            "operating_cost": 0.001,
            "replacement_cost": 3.0,
        }


class TestCostTypeFeatureMatrices:
    """Feature matrix shapes and values for each cost type."""

    @pytest.fixture(params=VALID_COST_TYPES)
    def cost_type(self, request):
        return request.param

    def _make_env(self, cost_type, n=20):
        if cost_type == "quadratic":
            return RustBusEnvironment(
                cost_type="quadratic",
                operating_cost_params=(0.01, 0.0001),
                replacement_cost=2.0,
                num_mileage_bins=n,
                discount_factor=0.99,
            )
        elif cost_type == "cubic":
            return RustBusEnvironment(
                cost_type="cubic",
                operating_cost_params=(0.01, 0.0001, 0.000001),
                replacement_cost=2.0,
                num_mileage_bins=n,
                discount_factor=0.99,
            )
        else:
            return RustBusEnvironment(
                cost_type=cost_type,
                operating_cost=0.01,
                replacement_cost=2.0,
                num_mileage_bins=n,
                discount_factor=0.99,
            )

    def test_feature_matrix_shape(self, cost_type):
        expected_n_features = {
            "linear": 2, "quadratic": 3, "cubic": 4, "sqrt": 2, "hyperbolic": 2,
        }
        env = self._make_env(cost_type)
        n_features = expected_n_features[cost_type]
        assert env.feature_matrix.shape == (20, 2, n_features)

    def test_replace_features_last_column(self, cost_type):
        """Replace action features should be [0, ..., 0, -1]."""
        env = self._make_env(cost_type)
        n_features = env.feature_matrix.shape[2]
        for s in range(env.num_states):
            # All feature columns except last should be 0 for replace
            for k in range(n_features - 1):
                assert float(env.feature_matrix[s, 1, k]) == 0.0
            # Last column is -1 for replace
            assert float(env.feature_matrix[s, 1, -1]) == -1.0

    def test_feature_matrix_reproduces_flow_utility(self, cost_type):
        """U(s,a) = theta dot phi(s,a) should match _compute_flow_utility."""
        env = self._make_env(cost_type)
        theta = env.get_true_parameter_vector()
        features = env.feature_matrix
        for s in range(env.num_states):
            for a in range(2):
                utility_from_features = float(jnp.dot(features[s, a], theta))
                utility_from_method = env._compute_flow_utility(s, a)
                np.testing.assert_allclose(
                    utility_from_features, utility_from_method, atol=1e-5,
                    err_msg=f"cost_type={cost_type}, state={s}, action={a}",
                )

    def test_parameter_count(self, cost_type):
        expected = {
            "linear": 2, "quadratic": 3, "cubic": 4, "sqrt": 2, "hyperbolic": 2,
        }
        env = self._make_env(cost_type)
        assert len(env.parameter_names) == expected[cost_type]
        assert len(env.true_parameters) == expected[cost_type]
        assert len(env.get_true_parameter_vector()) == expected[cost_type]


class TestCostTypeValidation:
    """Validation and error handling for cost_type."""

    def test_invalid_cost_type_raises(self):
        with pytest.raises(ValueError, match="cost_type must be one of"):
            RustBusEnvironment(cost_type="exponential")

    def test_wrong_param_count_quadratic(self):
        with pytest.raises(ValueError, match="expects 2"):
            RustBusEnvironment(
                cost_type="quadratic",
                operating_cost_params=(0.01,),
            )

    def test_wrong_param_count_cubic(self):
        with pytest.raises(ValueError, match="expects 3"):
            RustBusEnvironment(
                cost_type="cubic",
                operating_cost_params=(0.01, 0.001),
            )


class TestCostTypeSpecificValues:
    """Verify specific cost function formulas."""

    def test_sqrt_at_state_4(self):
        env = RustBusEnvironment(
            cost_type="sqrt", operating_cost=2.0,
            num_mileage_bins=10, discount_factor=0.99,
        )
        # U(4, keep) = -2.0 * sqrt(4) = -4.0
        assert abs(env._compute_flow_utility(4, 0) - (-4.0)) < 1e-10

    def test_hyperbolic_at_state_0(self):
        n = 10
        env = RustBusEnvironment(
            cost_type="hyperbolic", operating_cost=1.0,
            num_mileage_bins=n, discount_factor=0.99,
        )
        # U(0, keep) = -1.0 / (10+1 - 0) = -1/11
        expected = -1.0 / (n + 1)
        assert abs(env._compute_flow_utility(0, 0) - expected) < 1e-10

    def test_quadratic_at_state_3(self):
        env = RustBusEnvironment(
            cost_type="quadratic",
            operating_cost_params=(0.1, 0.01),
            replacement_cost=2.0,
            num_mileage_bins=10, discount_factor=0.99,
        )
        # U(3, keep) = -0.1*3 - 0.01*9 = -0.3 - 0.09 = -0.39
        expected = -0.1 * 3 - 0.01 * 9
        assert abs(env._compute_flow_utility(3, 0) - expected) < 1e-10

    def test_describe_shows_cost_type(self):
        env = RustBusEnvironment(cost_type="sqrt")
        desc = env.describe()
        assert "sqrt" in desc
