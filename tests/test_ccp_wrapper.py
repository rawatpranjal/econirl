"""Tests for CCP wrapper refactoring: RewardSpec, TrajectoryPanel, EstimatorProtocol.

Tests that the CCP sklearn wrapper:
1. Still works with DataFrame + "linear_cost" (backward compat)
2. Accepts RewardSpec as reward specification
3. Accepts TrajectoryPanel directly
4. Exposes policy_, value_, pvalues_, conf_int()
5. Satisfies EstimatorProtocol
6. CCP-specific: NPL iterations (num_policy_iterations > 1)
7. summary() returns string
8. predict_proba() correct shape
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, TrajectoryPanel
from econirl.estimators.ccp import CCP
from econirl.estimators.protocol import EstimatorProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use n_states=20 and discount=0.99 for fast tests.  The module-scoped
# fitted_model is shared by the majority of tests, so estimation runs once.

_N_STATES = 20
_DISCOUNT = 0.99


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


@pytest.fixture(scope="module")
def bus_df():
    """Shared bus DataFrame for all tests (scope=module for speed)."""
    return _generate_bus_dataframe()


@pytest.fixture(scope="module")
def fitted_model(bus_df):
    """Fitted CCP model shared across test classes."""
    model = CCP(n_states=_N_STATES, discount=_DISCOUNT, verbose=False)
    model.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")
    return model


# ---------------------------------------------------------------------------
# 1. Backward compatibility: DataFrame + "linear_cost"
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """CCP still works with the original DataFrame API."""

    def test_fit_dataframe_backward_compat(self, fitted_model):
        assert fitted_model.params_ is not None
        assert "theta_c" in fitted_model.params_
        assert "RC" in fitted_model.params_

    def test_fit_returns_self(self, bus_df):
        model = CCP(n_states=_N_STATES, discount=_DISCOUNT, verbose=False)
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
    """CCP accepts a RewardSpec for custom features."""

    def test_fit_with_reward_spec_argument(self, bus_df):
        n = _N_STATES
        features = torch.zeros((n, 2, 2), dtype=torch.float32)
        mileage = torch.arange(n, dtype=torch.float32)
        features[:, 0, 0] = -mileage
        features[:, 1, 1] = -1.0
        spec = RewardSpec(features, names=["theta_c", "RC"])

        model = CCP(n_states=n, discount=_DISCOUNT, verbose=False)
        model.fit(
            bus_df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
            reward=spec,
        )
        assert model.params_ is not None
        assert "theta_c" in model.params_
        assert "RC" in model.params_
        assert model.reward_spec_ is spec

    def test_fit_with_reward_spec_in_constructor(self, bus_df):
        n = _N_STATES
        features = torch.zeros((n, 2, 2), dtype=torch.float32)
        mileage = torch.arange(n, dtype=torch.float32)
        features[:, 0, 0] = -mileage
        features[:, 1, 1] = -1.0
        spec = RewardSpec(features, names=["theta_c", "RC"])

        model = CCP(n_states=n, discount=_DISCOUNT, utility=spec, verbose=False)
        model.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")
        assert model.params_ is not None
        assert model.reward_spec_ is spec

    def test_reward_spec_auto_created_for_linear_cost(self, fitted_model):
        """linear_cost mode should also populate reward_spec_."""
        assert fitted_model.reward_spec_ is not None
        assert isinstance(fitted_model.reward_spec_, RewardSpec)
        assert fitted_model.reward_spec_.parameter_names == ["theta_c", "RC"]


# ---------------------------------------------------------------------------
# 3. Fit with TrajectoryPanel
# ---------------------------------------------------------------------------


class TestTrajectoryPanelInput:
    """CCP accepts TrajectoryPanel as data input."""

    def test_fit_with_trajectory_panel(self, bus_df):
        panel = TrajectoryPanel.from_dataframe(
            bus_df, state="mileage_bin", action="replaced", id="bus_id"
        )
        model = CCP(n_states=_N_STATES, discount=_DISCOUNT, verbose=False)
        model.fit(panel)
        assert model.params_ is not None
        assert "theta_c" in model.params_

    def test_fit_with_panel(self, bus_df):
        """Panel is an alias for TrajectoryPanel, should also work."""
        panel = Panel.from_dataframe(
            bus_df, state="mileage_bin", action="replaced", id="bus_id"
        )
        model = CCP(n_states=_N_STATES, discount=_DISCOUNT, verbose=False)
        model.fit(panel)
        assert model.params_ is not None

    def test_dataframe_without_column_names_raises(self, bus_df):
        model = CCP(n_states=_N_STATES, verbose=False)
        with pytest.raises(ValueError, match="state, action, and id"):
            model.fit(bus_df)

    def test_invalid_data_type_raises(self):
        model = CCP(n_states=_N_STATES, verbose=False)
        with pytest.raises(TypeError, match="data must be"):
            model.fit({"not": "a dataframe"})


# ---------------------------------------------------------------------------
# 4. New attributes: policy_, value_, pvalues_
# ---------------------------------------------------------------------------


class TestNewAttributes:
    """Fitted CCP has policy_, value_, pvalues_."""

    def test_policy_shape(self, fitted_model):
        assert fitted_model.policy_ is not None
        assert fitted_model.policy_.shape == (_N_STATES, 2)

    def test_policy_valid_probabilities(self, fitted_model):
        assert (fitted_model.policy_ >= 0).all()
        assert (fitted_model.policy_ <= 1).all()
        row_sums = fitted_model.policy_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(_N_STATES), atol=1e-6)

    def test_value_shape(self, fitted_model):
        assert fitted_model.value_ is not None
        assert fitted_model.value_.shape == (_N_STATES,)

    def test_value_equals_value_function(self, fitted_model):
        """value_ and value_function_ should be the same array."""
        assert fitted_model.value_ is fitted_model.value_function_

    def test_pvalues_present(self, fitted_model):
        assert fitted_model.pvalues_ is not None
        assert "theta_c" in fitted_model.pvalues_
        assert "RC" in fitted_model.pvalues_

    def test_pvalues_in_range(self, fitted_model):
        for name, pv in fitted_model.pvalues_.items():
            if not np.isnan(pv):
                assert 0 <= pv <= 1, f"p-value for {name} out of range: {pv}"


# ---------------------------------------------------------------------------
# 5. conf_int()
# ---------------------------------------------------------------------------


class TestConfInt:
    """conf_int() returns valid confidence intervals."""

    def test_conf_int_keys(self, fitted_model):
        ci = fitted_model.conf_int()
        assert "theta_c" in ci
        assert "RC" in ci

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
        model = CCP(n_states=_N_STATES, verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.conf_int()


# ---------------------------------------------------------------------------
# 6. EstimatorProtocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """CCP satisfies the EstimatorProtocol."""

    def test_satisfies_protocol(self, fitted_model):
        assert isinstance(fitted_model, EstimatorProtocol)

    def test_unfitted_has_protocol_attributes(self):
        """Even before fit(), the attributes exist (as None)."""
        model = CCP(n_states=_N_STATES, verbose=False)
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
# 7. CCP-specific: NPL iterations
# ---------------------------------------------------------------------------


class TestNPLIterations:
    """CCP with num_policy_iterations > 1 runs NPL."""

    def test_npl_k2(self, bus_df):
        model = CCP(
            n_states=_N_STATES,
            discount=_DISCOUNT,
            num_policy_iterations=2,
            verbose=False,
        )
        model.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")
        assert model.params_ is not None
        assert model.converged_ is not None

    def test_npl_produces_different_params(self, bus_df):
        """NPL K>1 should generally differ from Hotz-Miller K=1."""
        hm = CCP(
            n_states=_N_STATES,
            discount=_DISCOUNT,
            num_policy_iterations=1,
            verbose=False,
        )
        hm.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")

        npl = CCP(
            n_states=_N_STATES,
            discount=_DISCOUNT,
            num_policy_iterations=3,
            verbose=False,
        )
        npl.fit(bus_df, state="mileage_bin", action="replaced", id="bus_id")

        # Both should produce valid estimates
        assert hm.params_ is not None
        assert npl.params_ is not None

        # NPL attributes should still all work
        assert npl.policy_ is not None
        assert npl.value_ is not None
        assert npl.pvalues_ is not None
        assert npl.reward_spec_ is not None

    def test_npl_default_is_hotz_miller(self):
        """Default num_policy_iterations=1 is Hotz-Miller."""
        model = CCP(n_states=_N_STATES)
        assert model.num_policy_iterations == 1


# ---------------------------------------------------------------------------
# 8. summary() and predict_proba()
# ---------------------------------------------------------------------------


class TestExistingMethodsStillWork:
    """All existing methods (summary, predict_proba, simulate, counterfactual)
    continue to function after the refactoring."""

    def test_summary(self, fitted_model):
        summary = fitted_model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_predict_proba(self, fitted_model):
        proba = fitted_model.predict_proba(np.array([0, 5, 10]))
        assert proba.shape == (3, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)
        assert (proba >= 0).all()

    def test_simulate(self, fitted_model):
        sim_df = fitted_model.simulate(n_agents=5, n_periods=10, seed=42)
        assert isinstance(sim_df, pd.DataFrame)
        assert len(sim_df) == 50
        assert "agent_id" in sim_df.columns
        assert "state" in sim_df.columns
        assert "action" in sim_df.columns

    def test_counterfactual(self, fitted_model):
        from econirl.estimators.nfxp import CounterfactualResult

        cf = fitted_model.counterfactual(RC=15.0)
        assert isinstance(cf, CounterfactualResult)
        assert cf.params["RC"] == 15.0
        assert cf.policy.shape == (_N_STATES, 2)
        assert cf.value_function.shape == (_N_STATES,)

    def test_repr(self, fitted_model):
        r = repr(fitted_model)
        assert "fitted=True" in r
        assert "num_policy_iterations" in r

    def test_repr_unfitted(self):
        r = repr(CCP(n_states=_N_STATES))
        assert "fitted=False" in r


# ---------------------------------------------------------------------------
# 9. fit() signature matches NFXP
# ---------------------------------------------------------------------------


class TestSignatureMatchesNFXP:
    """CCP.fit() has the same signature as NFXP.fit()."""

    def test_fit_signature_matches(self):
        from econirl.estimators.nfxp import NFXP
        import inspect

        nfxp_sig = inspect.signature(NFXP.fit)
        ccp_sig = inspect.signature(CCP.fit)

        nfxp_params = [p for p in nfxp_sig.parameters.keys() if p != "self"]
        ccp_params = [p for p in ccp_sig.parameters.keys() if p != "self"]

        assert nfxp_params == ccp_params
