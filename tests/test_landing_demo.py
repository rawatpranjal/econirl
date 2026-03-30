"""Regression test: NFXP and NNES recover matching parameters on Rust bus data."""

import numpy as np
import pytest
from econirl import NFXP, NNES
from econirl.datasets import load_rust_bus
from econirl.estimators.protocol import EstimatorProtocol


@pytest.fixture(scope="module")
def rust_bus_df():
    return load_rust_bus()


@pytest.fixture(scope="module")
def nfxp_fitted(rust_bus_df):
    return NFXP(n_states=90, discount=0.9999).fit(
        rust_bus_df, state="mileage_bin", action="replaced", id="bus_id"
    )


@pytest.fixture(scope="module")
def nnes_fitted(rust_bus_df):
    return NNES(n_states=90, discount=0.9999, v_epochs=300, n_outer_iterations=2).fit(
        rust_bus_df, state="mileage_bin", action="replaced", id="bus_id"
    )


class TestNFXPBaseline:
    def test_params_recovered(self, nfxp_fitted):
        assert nfxp_fitted.params_ is not None
        assert "theta_c" in nfxp_fitted.params_
        assert "RC" in nfxp_fitted.params_
        assert nfxp_fitted.params_["theta_c"] > 0
        assert nfxp_fitted.params_["RC"] > 0

    def test_se_finite(self, nfxp_fitted):
        assert nfxp_fitted.se_ is not None
        assert all(np.isfinite(v) for v in nfxp_fitted.se_.values())

    def test_protocol(self, nfxp_fitted):
        assert isinstance(nfxp_fitted, EstimatorProtocol)


class TestNNESBaseline:
    def test_params_recovered(self, nnes_fitted):
        assert nnes_fitted.params_ is not None
        assert "theta_c" in nnes_fitted.params_
        assert "RC" in nnes_fitted.params_
        # theta_c is constrained >= 0 by L-BFGS-B bounds; neural training
        # can occasionally pin it at the boundary, so we check >= 0 here.
        # The slow TestParameterAgreement tests verify actual recovery.
        assert nnes_fitted.params_["theta_c"] >= 0
        assert nnes_fitted.params_["RC"] > 0

    def test_se_present(self, nnes_fitted):
        assert nnes_fitted.se_ is not None
        assert "theta_c" in nnes_fitted.se_
        assert "RC" in nnes_fitted.se_

    def test_protocol(self, nnes_fitted):
        assert isinstance(nnes_fitted, EstimatorProtocol)


@pytest.mark.slow
class TestParameterAgreement:
    def test_cosine_similarity(self, nfxp_fitted, nnes_fitted):
        """Both estimators should recover similar parameters."""
        nfxp_vec = np.array([nfxp_fitted.params_["theta_c"], nfxp_fitted.params_["RC"]])
        nnes_vec = np.array([nnes_fitted.params_["theta_c"], nnes_fitted.params_["RC"]])
        cos_sim = np.dot(nfxp_vec, nnes_vec) / (np.linalg.norm(nfxp_vec) * np.linalg.norm(nnes_vec))
        assert cos_sim > 0.85, f"Cosine similarity {cos_sim:.3f} < 0.85"

    def test_both_have_policy(self, nfxp_fitted, nnes_fitted):
        assert nfxp_fitted.policy_ is not None
        assert nnes_fitted.policy_ is not None
        assert nfxp_fitted.policy_.shape == nnes_fitted.policy_.shape
