"""Tests for SEES sklearn-style wrapper.

Tests that the SEES estimator:
1. Fits with DataFrame + "linear_cost" (basic functionality)
2. Recovers positive parameters (theta_c > 0, RC > 0)
3. Exposes policy_, value_, pvalues_ attributes
4. conf_int() returns valid intervals
5. Accepts RewardSpec as reward specification
6. Exposes alpha_ attribute after fit (basis coefficients)
7. summary() returns non-empty string
8. predict_proba() returns correct shape and valid probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, TrajectoryPanel
from econirl.estimators.sees import SEES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_STATES_FAST = 5
_DISCOUNT_FAST = 0.99


def _generate_bus_dataframe(
    n_individuals: int = 15,
    n_periods: int = 25,
    n_states: int = _N_STATES_FAST,
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


@pytest.fixture(scope="module")
def bus_df_fast():
    """Shared small bus DataFrame for fast tests."""
    return _generate_bus_dataframe(n_states=_N_STATES_FAST)


@pytest.fixture(scope="module")
def fitted_model_fast(bus_df_fast):
    """Fitted SEES model with small basis for speed."""
    model = SEES(
        n_states=_N_STATES_FAST,
        discount=_DISCOUNT_FAST,
        basis_type="fourier",
        basis_dim=4,
        penalty_lambda=0.01,
        max_iter=100,
        verbose=False,
    )
    model.fit(bus_df_fast, state="mileage_bin", action="replaced", id="bus_id")
    return model


# ---------------------------------------------------------------------------
# 1. Basic fit
# ---------------------------------------------------------------------------


class TestBasicFit:
    """SEES fits with DataFrame + linear_cost."""

    def test_fit_produces_params(self, fitted_model_fast):
        assert fitted_model_fast.params_ is not None
        assert "theta_c" in fitted_model_fast.params_
        assert "RC" in fitted_model_fast.params_

    def test_fit_returns_self(self, bus_df_fast):
        model = SEES(
            n_states=_N_STATES_FAST,
            discount=_DISCOUNT_FAST,
            basis_dim=4,
            max_iter=50,
            verbose=False,
        )
        result = model.fit(
            bus_df_fast, state="mileage_bin", action="replaced", id="bus_id"
        )
        assert result is model

    def test_coef_array(self, fitted_model_fast):
        assert fitted_model_fast.coef_ is not None
        assert isinstance(fitted_model_fast.coef_, np.ndarray)
        assert len(fitted_model_fast.coef_) == 2

    def test_log_likelihood(self, fitted_model_fast):
        assert fitted_model_fast.log_likelihood_ is not None
        assert np.isfinite(fitted_model_fast.log_likelihood_)

    def test_converged(self, fitted_model_fast):
        assert fitted_model_fast.converged_ is not None


# ---------------------------------------------------------------------------
# 2. Parameters recovered (positive theta_c and RC)
# ---------------------------------------------------------------------------


class TestParametersRecovered:
    """Estimated parameters are positive (basic sanity)."""

    def test_theta_c_positive(self, fitted_model_fast):
        assert fitted_model_fast.params_["theta_c"] > 0

    def test_RC_positive(self, fitted_model_fast):
        assert fitted_model_fast.params_["RC"] > 0


# ---------------------------------------------------------------------------
# 3. Attributes: policy_, value_, pvalues_
# ---------------------------------------------------------------------------


class TestAttributes:
    """Fitted SEES has policy_, value_, pvalues_."""

    def test_policy_shape(self, fitted_model_fast):
        assert fitted_model_fast.policy_ is not None
        assert fitted_model_fast.policy_.shape == (_N_STATES_FAST, 2)

    def test_policy_valid_probabilities(self, fitted_model_fast):
        assert (fitted_model_fast.policy_ >= 0).all()
        assert (fitted_model_fast.policy_ <= 1).all()
        row_sums = fitted_model_fast.policy_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(_N_STATES_FAST), atol=1e-6)

    def test_value_shape(self, fitted_model_fast):
        assert fitted_model_fast.value_ is not None
        assert fitted_model_fast.value_.shape == (_N_STATES_FAST,)

    def test_pvalues_present(self, fitted_model_fast):
        assert fitted_model_fast.pvalues_ is not None
        assert "theta_c" in fitted_model_fast.pvalues_
        assert "RC" in fitted_model_fast.pvalues_

    def test_se_present(self, fitted_model_fast):
        assert fitted_model_fast.se_ is not None
        assert "theta_c" in fitted_model_fast.se_
        assert "RC" in fitted_model_fast.se_

    def test_transitions_estimated(self, fitted_model_fast):
        assert fitted_model_fast.transitions_ is not None


# ---------------------------------------------------------------------------
# 4. conf_int()
# ---------------------------------------------------------------------------


class TestConfInt:
    """conf_int() returns valid confidence intervals."""

    def test_conf_int_keys(self, fitted_model_fast):
        ci = fitted_model_fast.conf_int()
        assert "theta_c" in ci
        assert "RC" in ci

    def test_conf_int_brackets_estimate(self, fitted_model_fast):
        ci = fitted_model_fast.conf_int()
        for name in fitted_model_fast.params_:
            lower, upper = ci[name]
            est = fitted_model_fast.params_[name]
            if np.isnan(lower) or np.isnan(upper):
                continue
            assert lower <= est <= upper, (
                f"CI for {name}: ({lower}, {upper}) does not contain "
                f"estimate {est}"
            )

    def test_conf_int_unfitted_raises(self):
        model = SEES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.conf_int()


# ---------------------------------------------------------------------------
# 5. RewardSpec input
# ---------------------------------------------------------------------------


class TestRewardSpec:
    """SEES accepts a RewardSpec for custom features."""

    def test_fit_with_reward_spec_argument(self, bus_df_fast):
        import jax.numpy as jnp

        n = _N_STATES_FAST
        features = jnp.zeros((n, 2, 2))
        mileage = jnp.arange(n, dtype=jnp.float32)
        features = features.at[:, 0, 0].set(-mileage)
        features = features.at[:, 1, 1].set(-1.0)
        spec = RewardSpec(features, names=["theta_c", "RC"])

        model = SEES(
            n_states=n,
            discount=_DISCOUNT_FAST,
            basis_dim=4,
            max_iter=50,
            verbose=False,
        )
        model.fit(
            bus_df_fast,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
            reward=spec,
        )
        assert model.params_ is not None
        assert "theta_c" in model.params_


# ---------------------------------------------------------------------------
# 6. alpha_ attribute (basis coefficients)
# ---------------------------------------------------------------------------


class TestAlpha:
    """Basis coefficients are accessible after fit."""

    def test_alpha_attribute_exists(self, fitted_model_fast):
        assert hasattr(fitted_model_fast, "alpha_")

    def test_alpha_shape(self, fitted_model_fast):
        if fitted_model_fast.alpha_ is not None:
            assert fitted_model_fast.alpha_.shape == (4,)

    def test_alpha_finite(self, fitted_model_fast):
        if fitted_model_fast.alpha_ is not None:
            assert np.all(np.isfinite(fitted_model_fast.alpha_))


# ---------------------------------------------------------------------------
# 7. summary()
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a non-empty string."""

    def test_summary_non_empty(self, fitted_model_fast):
        s = fitted_model_fast.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_unfitted(self):
        model = SEES(n_states=_N_STATES_FAST, verbose=False)
        s = model.summary()
        assert "Not fitted" in s


# ---------------------------------------------------------------------------
# 8. predict_proba()
# ---------------------------------------------------------------------------


class TestPredictProba:
    """predict_proba() returns correct shape and valid probabilities."""

    def test_predict_proba_shape(self, fitted_model_fast):
        proba = fitted_model_fast.predict_proba(np.array([0, 1, 2]))
        assert proba.shape == (3, 2)

    def test_predict_proba_valid(self, fitted_model_fast):
        proba = fitted_model_fast.predict_proba(np.array([0, 1, 2]))
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)

    def test_predict_proba_unfitted_raises(self):
        model = SEES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0]))


# ---------------------------------------------------------------------------
# 9. Data input variants
# ---------------------------------------------------------------------------


class TestDataInputs:
    """SEES accepts different data input types."""

    def test_fit_with_trajectory_panel(self, bus_df_fast):
        panel = TrajectoryPanel.from_dataframe(
            bus_df_fast, state="mileage_bin", action="replaced", id="bus_id"
        )
        model = SEES(
            n_states=_N_STATES_FAST,
            discount=_DISCOUNT_FAST,
            basis_dim=4,
            max_iter=50,
            verbose=False,
        )
        model.fit(panel)
        assert model.params_ is not None

    def test_dataframe_without_column_names_raises(self, bus_df_fast):
        model = SEES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(ValueError, match="state, action, and id"):
            model.fit(bus_df_fast)

    def test_invalid_data_type_raises(self):
        model = SEES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(TypeError, match="data must be"):
            model.fit({"not": "a dataframe"})


# ---------------------------------------------------------------------------
# 10. repr
# ---------------------------------------------------------------------------


class TestRepr:
    """SEES repr works before and after fitting."""

    def test_repr_fitted(self, fitted_model_fast):
        r = repr(fitted_model_fast)
        assert "fitted=True" in r
        assert "SEES" in r

    def test_repr_unfitted(self):
        r = repr(SEES(n_states=_N_STATES_FAST))
        assert "fitted=False" in r
        assert "SEES" in r
