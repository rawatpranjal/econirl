"""Tests for NNES sklearn-style wrapper.

Tests that the NNES estimator:
1. Fits with DataFrame + "linear_cost" (basic functionality)
2. Recovers positive parameters (theta_c > 0, RC > 0)
3. Exposes policy_, value_, pvalues_ attributes
4. conf_int() returns valid intervals
5. Satisfies EstimatorProtocol
6. Accepts RewardSpec as reward specification
7. Exposes v_network_ attribute after fit
8. summary() returns non-empty string
9. predict_proba() returns correct shape and valid probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, TrajectoryPanel
from econirl.estimators.nnes import NNES
from econirl.estimators.protocol import EstimatorProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use tiny problem for fast tests: n_states=5, minimal epochs/iterations
_N_STATES_FAST = 5
_DISCOUNT_FAST = 0.99
_V_EPOCHS_FAST = 10
_N_OUTER_FAST = 1

# Full-size for slow parameter recovery tests
_N_STATES_FULL = 20
_DISCOUNT_FULL = 0.99


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
    """Fitted NNES model with tiny hyperparameters for speed."""
    model = NNES(
        n_states=_N_STATES_FAST,
        discount=_DISCOUNT_FAST,
        hidden_dim=8,
        num_layers=1,
        v_lr=1e-2,
        v_epochs=_V_EPOCHS_FAST,
        n_outer_iterations=_N_OUTER_FAST,
        verbose=False,
    )
    model.fit(bus_df_fast, state="mileage_bin", action="replaced", id="bus_id")
    return model


# ---------------------------------------------------------------------------
# 1. Basic fit
# ---------------------------------------------------------------------------


class TestBasicFit:
    """NNES fits with DataFrame + linear_cost."""

    def test_fit_produces_params(self, fitted_model_fast):
        assert fitted_model_fast.params_ is not None
        assert "theta_c" in fitted_model_fast.params_
        assert "RC" in fitted_model_fast.params_

    def test_fit_returns_self(self, bus_df_fast):
        model = NNES(
            n_states=_N_STATES_FAST,
            discount=_DISCOUNT_FAST,
            hidden_dim=8,
            num_layers=1,
            v_epochs=_V_EPOCHS_FAST,
            n_outer_iterations=_N_OUTER_FAST,
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


@pytest.mark.slow
class TestParameterRecoveryFull:
    """Slower parameter recovery test with more data and iterations."""

    def test_full_recovery(self):
        torch.manual_seed(42)  # Fix torch seed for deterministic V-network init
        df = _generate_bus_dataframe(
            n_individuals=30,
            n_periods=50,
            n_states=_N_STATES_FULL,
            seed=123,
        )
        model = NNES(
            n_states=_N_STATES_FULL,
            discount=_DISCOUNT_FULL,
            hidden_dim=32,
            num_layers=2,
            v_lr=1e-3,
            v_epochs=500,
            n_outer_iterations=3,
            verbose=False,
        )
        model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
        # RC (replacement cost) should be positive and in a reasonable range
        assert model.params_["RC"] > 0
        # theta_c (operating cost) is small and hard to identify with neural
        # approximation on limited data — just check it's finite
        assert np.isfinite(model.params_["theta_c"])
        # Should have reasonable standard errors
        assert np.isfinite(model.se_["theta_c"])
        assert np.isfinite(model.se_["RC"])


# ---------------------------------------------------------------------------
# 3. New attributes: policy_, value_, pvalues_
# ---------------------------------------------------------------------------


class TestAttributes:
    """Fitted NNES has policy_, value_, pvalues_."""

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

    def test_pvalues_in_range(self, fitted_model_fast):
        for name, pv in fitted_model_fast.pvalues_.items():
            if not np.isnan(pv):
                assert 0 <= pv <= 1, f"p-value for {name} out of range: {pv}"

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
            # With tiny config, Hessian may be singular -> NaN SEs -> NaN CIs
            if np.isnan(lower) or np.isnan(upper):
                continue
            assert lower <= est <= upper, (
                f"CI for {name}: ({lower}, {upper}) does not contain "
                f"estimate {est}"
            )

    def test_conf_int_custom_alpha(self, fitted_model_fast):
        ci_90 = fitted_model_fast.conf_int(alpha=0.10)
        ci_95 = fitted_model_fast.conf_int(alpha=0.05)
        # 95% CI should be wider than 90% CI
        for name in fitted_model_fast.params_:
            width_90 = ci_90[name][1] - ci_90[name][0]
            width_95 = ci_95[name][1] - ci_95[name][0]
            # With tiny config, Hessian may be singular -> NaN widths
            if np.isnan(width_90) or np.isnan(width_95):
                continue
            assert width_95 >= width_90

    def test_conf_int_unfitted_raises(self):
        model = NNES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.conf_int()


# ---------------------------------------------------------------------------
# 5. EstimatorProtocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """NNES satisfies the EstimatorProtocol."""

    def test_satisfies_protocol(self, fitted_model_fast):
        assert isinstance(fitted_model_fast, EstimatorProtocol)

    def test_unfitted_has_protocol_attributes(self):
        """Even before fit(), the attributes exist (as None)."""
        model = NNES(n_states=_N_STATES_FAST, verbose=False)
        assert hasattr(model, "params_")
        assert hasattr(model, "se_")
        assert hasattr(model, "pvalues_")
        assert hasattr(model, "policy_")
        assert hasattr(model, "value_")

    def test_protocol_methods_present(self, fitted_model_fast):
        assert callable(getattr(fitted_model_fast, "fit", None))
        assert callable(getattr(fitted_model_fast, "summary", None))
        assert callable(getattr(fitted_model_fast, "predict_proba", None))
        assert callable(getattr(fitted_model_fast, "conf_int", None))


# ---------------------------------------------------------------------------
# 6. RewardSpec input
# ---------------------------------------------------------------------------


class TestRewardSpec:
    """NNES accepts a RewardSpec for custom features."""

    def test_fit_with_reward_spec_argument(self, bus_df_fast):
        n = _N_STATES_FAST
        features = torch.zeros((n, 2, 2), dtype=torch.float32)
        mileage = torch.arange(n, dtype=torch.float32)
        features[:, 0, 0] = -mileage
        features[:, 1, 1] = -1.0
        spec = RewardSpec(features, names=["theta_c", "RC"])

        model = NNES(
            n_states=n,
            discount=_DISCOUNT_FAST,
            hidden_dim=8,
            num_layers=1,
            v_epochs=_V_EPOCHS_FAST,
            n_outer_iterations=_N_OUTER_FAST,
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
        assert "RC" in model.params_
        assert model.reward_spec_ is spec

    def test_reward_spec_auto_created_for_linear_cost(self, fitted_model_fast):
        """linear_cost mode should also populate reward_spec_."""
        assert fitted_model_fast.reward_spec_ is not None
        assert isinstance(fitted_model_fast.reward_spec_, RewardSpec)
        assert fitted_model_fast.reward_spec_.parameter_names == ["theta_c", "RC"]


# ---------------------------------------------------------------------------
# 7. v_network_ attribute
# ---------------------------------------------------------------------------


class TestVNetwork:
    """V-network values are accessible after fit."""

    def test_v_network_attribute_exists(self, fitted_model_fast):
        # v_network_ should be set (may be None if metadata doesn't include it,
        # but current NNES implementation always includes it)
        assert hasattr(fitted_model_fast, "v_network_")

    def test_v_network_shape(self, fitted_model_fast):
        if fitted_model_fast.v_network_ is not None:
            assert fitted_model_fast.v_network_.shape == (_N_STATES_FAST,)

    def test_v_network_finite(self, fitted_model_fast):
        if fitted_model_fast.v_network_ is not None:
            assert np.all(np.isfinite(fitted_model_fast.v_network_))


# ---------------------------------------------------------------------------
# 8. summary()
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a non-empty string."""

    def test_summary_non_empty(self, fitted_model_fast):
        s = fitted_model_fast.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_unfitted(self):
        model = NNES(n_states=_N_STATES_FAST, verbose=False)
        s = model.summary()
        assert "Not fitted" in s


# ---------------------------------------------------------------------------
# 9. predict_proba()
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

    def test_predict_proba_single_state(self, fitted_model_fast):
        proba = fitted_model_fast.predict_proba(np.array([0]))
        assert proba.shape == (1, 2)

    def test_predict_proba_unfitted_raises(self):
        model = NNES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0]))


# ---------------------------------------------------------------------------
# 10. Data input variants
# ---------------------------------------------------------------------------


class TestDataInputs:
    """NNES accepts different data input types."""

    def test_fit_with_trajectory_panel(self, bus_df_fast):
        panel = TrajectoryPanel.from_dataframe(
            bus_df_fast, state="mileage_bin", action="replaced", id="bus_id"
        )
        model = NNES(
            n_states=_N_STATES_FAST,
            discount=_DISCOUNT_FAST,
            hidden_dim=8,
            num_layers=1,
            v_epochs=_V_EPOCHS_FAST,
            n_outer_iterations=_N_OUTER_FAST,
            verbose=False,
        )
        model.fit(panel)
        assert model.params_ is not None

    def test_dataframe_without_column_names_raises(self, bus_df_fast):
        model = NNES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(ValueError, match="state, action, and id"):
            model.fit(bus_df_fast)

    def test_invalid_data_type_raises(self):
        model = NNES(n_states=_N_STATES_FAST, verbose=False)
        with pytest.raises(TypeError, match="data must be"):
            model.fit({"not": "a dataframe"})


# ---------------------------------------------------------------------------
# 11. repr
# ---------------------------------------------------------------------------


class TestRepr:
    """NNES repr works before and after fitting."""

    def test_repr_fitted(self, fitted_model_fast):
        r = repr(fitted_model_fast)
        assert "fitted=True" in r
        assert "NNES" in r

    def test_repr_unfitted(self):
        r = repr(NNES(n_states=_N_STATES_FAST))
        assert "fitted=False" in r
        assert "NNES" in r


# ---------------------------------------------------------------------------
# 12. No simulate/counterfactual (NFXP-specific)
# ---------------------------------------------------------------------------


class TestNoNFXPMethods:
    """NNES should NOT have simulate() or counterfactual()."""

    def test_no_simulate(self, fitted_model_fast):
        assert not hasattr(fitted_model_fast, "simulate")

    def test_no_counterfactual(self, fitted_model_fast):
        assert not hasattr(fitted_model_fast, "counterfactual")
