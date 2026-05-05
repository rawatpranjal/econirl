"""Tests for MCEIRL wrapper: RewardSpec, TrajectoryPanel, EstimatorProtocol.

Tests that the MCEIRL sklearn wrapper:
1. Still works with DataFrame (backward compat)
2. Accepts RewardSpec as reward specification
3. pvalues_ present and in [0, 1]
4. conf_int() returns valid intervals
5. value_ alias works
6. EstimatorProtocol conformance
7. reward_ attribute present (MCE-IRL specific)
8. summary() returns string
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, Trajectory, TrajectoryPanel
from econirl.estimators.mce_irl import MCEIRL
from econirl.estimators.protocol import EstimatorProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_STATES = 20
_DISCOUNT = 0.95


def _generate_bus_dataframe(
    n_individuals: int = 15,
    n_periods: int = 25,
    n_states: int = _N_STATES,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic bus replacement data."""
    np.random.seed(seed)
    data = []
    for i in range(n_individuals):
        state = 0
        for t in range(n_periods):
            action = (
                1 if state > n_states * 2 // 3 or np.random.random() < 0.05 else 0
            )
            next_state = (
                0
                if action == 1
                else min(
                    state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                    n_states - 1,
                )
            )
            data.append(
                {
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                }
            )
            state = next_state
    return pd.DataFrame(data)


def _make_features(n_states: int = _N_STATES) -> np.ndarray:
    """Create state features: linear and quadratic mileage cost."""
    s = np.arange(n_states)
    return np.column_stack([s / 100, (s / 100) ** 2])


@pytest.fixture(scope="module")
def bus_df():
    """Shared bus DataFrame for all tests (scope=module for speed)."""
    return _generate_bus_dataframe()


@pytest.fixture(scope="module")
def fitted_model(bus_df):
    """Fitted MCEIRL model shared across test classes."""
    features = _make_features()
    model = MCEIRL(
        n_states=_N_STATES,
        n_actions=2,
        discount=_DISCOUNT,
        feature_matrix=features,
        feature_names=["linear", "quadratic"],
        se_method="hessian",
        n_bootstrap=0,
        verbose=False,
    )
    model.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")
    return model


# ---------------------------------------------------------------------------
# 1. Backward compatibility: DataFrame
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """MCEIRL still works with the original DataFrame API."""

    def test_fit_dataframe_backward_compat(self, fitted_model):
        assert fitted_model.params_ is not None
        assert "linear" in fitted_model.params_
        assert "quadratic" in fitted_model.params_

    def test_fit_returns_self(self, bus_df):
        features = _make_features()
        model = MCEIRL(
            n_states=_N_STATES,
            discount=_DISCOUNT,
            feature_matrix=features,
            feature_names=["linear", "quadratic"],
            se_method="hessian",
            n_bootstrap=0,
            verbose=False,
        )
        result = model.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")
        assert result is model

    def test_existing_attributes_present(self, fitted_model):
        assert fitted_model.coef_ is not None
        assert isinstance(fitted_model.coef_, np.ndarray)
        assert fitted_model.log_likelihood_ is not None
        assert fitted_model.value_function_ is not None
        assert fitted_model.transitions_ is not None
        assert fitted_model.converged_ is not None
        assert fitted_model.se_ is not None


# ---------------------------------------------------------------------------
# 2. Fit with RewardSpec
# ---------------------------------------------------------------------------


class TestRewardSpec:
    """MCEIRL accepts a RewardSpec for custom features."""

    def test_fit_with_reward_spec_argument(self, bus_df):
        n = _N_STATES
        s = jnp.arange(n, dtype=jnp.float32)
        state_features = jnp.stack([s / 100, (s / 100) ** 2], axis=1)
        spec = RewardSpec(state_features, names=["linear", "quadratic"], n_actions=2)

        model = MCEIRL(
            n_states=n,
            discount=_DISCOUNT,
            se_method="hessian",
            n_bootstrap=0,
            verbose=False,
        )
        model.fit(
            bus_df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
            reward=spec,
        )
        assert model.params_ is not None
        assert "linear" in model.params_
        assert "quadratic" in model.params_
        assert model.reward_spec_ is spec

    def test_fit_without_reward_spec_or_features_raises(self, bus_df):
        model = MCEIRL(
            n_states=_N_STATES,
            n_actions=2,
            discount=_DISCOUNT,
            se_method="hessian",
            n_bootstrap=0,
            verbose=False,
        )
        with pytest.raises(ValueError, match="explicit reward specification"):
            model.fit(
                bus_df,
                state="mileage_bin",
                action="replaced",
                id="bus_id",
            )


# ---------------------------------------------------------------------------
# 3. pvalues_
# ---------------------------------------------------------------------------


class TestPValues:
    """pvalues_ present and in [0, 1]."""

    def test_pvalues_present(self, fitted_model):
        assert fitted_model.pvalues_ is not None
        assert "linear" in fitted_model.pvalues_
        assert "quadratic" in fitted_model.pvalues_

    def test_pvalues_in_range(self, fitted_model):
        for name, pv in fitted_model.pvalues_.items():
            if not np.isnan(pv):
                assert 0 <= pv <= 1, f"p-value for {name} out of range: {pv}"


# ---------------------------------------------------------------------------
# 4. conf_int()
# ---------------------------------------------------------------------------


class TestConfInt:
    """conf_int() returns valid confidence intervals."""

    def test_conf_int_keys(self, fitted_model):
        ci = fitted_model.conf_int()
        assert "linear" in ci
        assert "quadratic" in ci

    def test_conf_int_brackets_estimate(self, fitted_model):
        ci = fitted_model.conf_int()
        for name in fitted_model.params_:
            lower, upper = ci[name]
            est = fitted_model.params_[name]
            assert lower <= est <= upper, (
                f"CI for {name}: ({lower}, {upper}) does not contain "
                f"estimate {est}"
            )

    def test_conf_int_custom_alpha(self, fitted_model):
        ci_90 = fitted_model.conf_int(alpha=0.10)
        ci_95 = fitted_model.conf_int(alpha=0.05)
        # 95% CI should be wider than 90% CI
        for name in fitted_model.params_:
            width_90 = ci_90[name][1] - ci_90[name][0]
            width_95 = ci_95[name][1] - ci_95[name][0]
            assert width_95 >= width_90

    def test_conf_int_unfitted_raises(self):
        model = MCEIRL(n_states=_N_STATES, verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.conf_int()


# ---------------------------------------------------------------------------
# 5. value_ alias
# ---------------------------------------------------------------------------


class TestValueAlias:
    """value_ alias for value_function_."""

    def test_value_shape(self, fitted_model):
        assert fitted_model.value_ is not None
        assert fitted_model.value_.shape == (_N_STATES,)

    def test_value_equals_value_function(self, fitted_model):
        """value_ and value_function_ should be the same array."""
        assert fitted_model.value_ is fitted_model.value_function_


# ---------------------------------------------------------------------------
# 6. EstimatorProtocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """MCEIRL satisfies the EstimatorProtocol."""

    def test_satisfies_protocol(self, fitted_model):
        assert isinstance(fitted_model, EstimatorProtocol)

    def test_unfitted_has_protocol_attributes(self):
        """Even before fit(), the attributes exist (as None)."""
        model = MCEIRL(n_states=_N_STATES, verbose=False)
        assert hasattr(model, "params_")
        assert hasattr(model, "se_")
        assert hasattr(model, "pvalues_")
        assert hasattr(model, "policy_")
        assert hasattr(model, "value_")

    def test_protocol_methods_present(self, fitted_model):
        assert callable(getattr(fitted_model, "fit", None))
        assert callable(getattr(fitted_model, "summary", None))
        assert callable(getattr(fitted_model, "predict_proba", None))
        assert callable(getattr(fitted_model, "conf_int", None))


# ---------------------------------------------------------------------------
# 7. reward_ attribute (MCE-IRL specific)
# ---------------------------------------------------------------------------


class TestRewardAttribute:
    """reward_ attribute present after fitting."""

    def test_reward_present(self, fitted_model):
        assert fitted_model.reward_ is not None
        assert isinstance(fitted_model.reward_, np.ndarray)
        assert fitted_model.reward_.shape == (_N_STATES,)


# ---------------------------------------------------------------------------
# 8. summary() returns string
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a non-empty string."""

    def test_summary_returns_string(self, fitted_model):
        summary = fitted_model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "MCE IRL" in summary or "Maximum Causal Entropy" in summary

    def test_summary_unfitted(self):
        model = MCEIRL(n_states=_N_STATES, verbose=False)
        summary = model.summary()
        assert isinstance(summary, str)
        assert "Not fitted" in summary

    def test_repr(self, fitted_model):
        r = repr(fitted_model)
        assert "MCEIRL" in r
        assert "fitted=True" in r

    def test_repr_unfitted(self):
        r = repr(MCEIRL(n_states=_N_STATES))
        assert "fitted=False" in r
