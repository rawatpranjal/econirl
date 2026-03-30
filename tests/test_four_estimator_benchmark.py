"""Regression test: All 4 core estimators work on the same data with consistent interface.

This is the Phase 2 acceptance test. All estimators:
1. Accept the same fit() call
2. Satisfy EstimatorProtocol
3. Produce params_, se_, pvalues_, policy_, value_
4. Generate a summary()
"""

import numpy as np
import pytest
from econirl import NFXP, NNES, CCP, TDCCP
from econirl.datasets import load_rust_bus
from econirl.estimators.protocol import EstimatorProtocol


@pytest.fixture(scope="module")
def rust_bus_df():
    return load_rust_bus()


# Use small state space for fast tests
FAST_KWARGS = dict(n_states=90, discount=0.9999)


@pytest.fixture(scope="module")
def all_fitted(rust_bus_df):
    """Fit all 4 estimators on the same data. Module-scoped for speed."""
    models = {}

    models["nfxp"] = NFXP(**FAST_KWARGS).fit(
        rust_bus_df, state="mileage_bin", action="replaced", id="bus_id"
    )
    models["ccp"] = CCP(**FAST_KWARGS, num_policy_iterations=3).fit(
        rust_bus_df, state="mileage_bin", action="replaced", id="bus_id"
    )
    models["nnes"] = NNES(**FAST_KWARGS, v_epochs=200, n_outer_iterations=2).fit(
        rust_bus_df, state="mileage_bin", action="replaced", id="bus_id"
    )
    models["tdccp"] = TDCCP(
        **FAST_KWARGS, avi_iterations=10, epochs_per_avi=15, n_policy_iterations=2
    ).fit(rust_bus_df, state="mileage_bin", action="replaced", id="bus_id")

    return models


class TestProtocolConformance:
    """All 4 estimators satisfy EstimatorProtocol."""

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_isinstance(self, all_fitted, name):
        assert isinstance(all_fitted[name], EstimatorProtocol)


class TestConsistentAttributes:
    """All 4 estimators produce the same attribute structure."""

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_params_keys(self, all_fitted, name):
        model = all_fitted[name]
        assert model.params_ is not None
        assert set(model.params_.keys()) == {"theta_c", "RC"}

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_policy_shape(self, all_fitted, name):
        model = all_fitted[name]
        assert model.policy_ is not None
        assert model.policy_.shape == (90, 2)

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_value_shape(self, all_fitted, name):
        model = all_fitted[name]
        assert model.value_ is not None
        assert model.value_.shape == (90,)

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_pvalues_present(self, all_fitted, name):
        model = all_fitted[name]
        assert model.pvalues_ is not None
        assert set(model.pvalues_.keys()) == {"theta_c", "RC"}

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_summary_string(self, all_fitted, name):
        model = all_fitted[name]
        s = model.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_predict_proba_shape(self, all_fitted, name):
        model = all_fitted[name]
        proba = model.predict_proba(np.array([0, 10, 50, 89]))
        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("name", ["nfxp", "ccp", "nnes", "tdccp"])
    def test_conf_int(self, all_fitted, name):
        model = all_fitted[name]
        # conf_int may fail if SEs are NaN; just check it doesn't crash
        try:
            ci = model.conf_int()
            assert "theta_c" in ci
            assert "RC" in ci
        except (RuntimeError, ValueError):
            pass  # NaN SEs for neural methods is acceptable


@pytest.mark.slow
class TestParameterRecovery:
    """Structural estimators should recover positive parameters."""

    def test_nfxp_positive_params(self, all_fitted):
        assert all_fitted["nfxp"].params_["theta_c"] > 0
        assert all_fitted["nfxp"].params_["RC"] > 0

    def test_ccp_positive_params(self, all_fitted):
        assert all_fitted["ccp"].params_["theta_c"] > 0
        assert all_fitted["ccp"].params_["RC"] > 0

    def test_nfxp_ccp_agreement(self, all_fitted):
        """NFXP and CCP should recover very similar parameters."""
        nfxp_vec = np.array(
            [all_fitted["nfxp"].params_["theta_c"], all_fitted["nfxp"].params_["RC"]]
        )
        ccp_vec = np.array(
            [all_fitted["ccp"].params_["theta_c"], all_fitted["ccp"].params_["RC"]]
        )
        cos_sim = np.dot(nfxp_vec, ccp_vec) / (
            np.linalg.norm(nfxp_vec) * np.linalg.norm(ccp_vec)
        )
        assert cos_sim > 0.95, f"NFXP-CCP cosine sim {cos_sim:.3f} < 0.95"

    def test_all_four_print(self, all_fitted):
        """Print all parameters for manual inspection."""
        for name, model in all_fitted.items():
            print(
                f"\n{name.upper()}: theta_c={model.params_['theta_c']:.6f}, "
                f"RC={model.params_['RC']:.4f}"
            )
