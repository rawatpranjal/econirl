"""Tests for sklearn-style MCEIRL estimator.

Tests the MCEIRL estimator class which provides a scikit-learn style
interface for Maximum Causal Entropy Inverse Reinforcement Learning (Ziebart 2010).
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp

from econirl.core.types import Panel, Trajectory
from econirl.estimators.mce_irl import MCEIRL


def _default_feature_matrix(n_states: int = 20) -> np.ndarray:
    return (np.arange(n_states, dtype=float).reshape(-1, 1) / max(n_states - 1, 1))


@pytest.fixture
def mce_irl_sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    n_states = 20
    for i in range(8):
        state = int(rng.integers(0, 3))
        for t in range(10):
            action = int(state > 12 or rng.random() < 0.08)
            rows.append({"id": i, "period": t, "state": state, "action": action})
            if action:
                state = 0
            else:
                state = min(state + int(rng.choice([0, 1, 2], p=[0.2, 0.7, 0.1])), n_states - 1)
    return pd.DataFrame(rows)


@pytest.fixture
def mce_irl_fitted_estimator(mce_irl_sample_df) -> MCEIRL:
    model = MCEIRL(
        n_states=20,
        discount=0.95,
        feature_matrix=_default_feature_matrix(20),
        feature_names=["f0"],
        verbose=False,
        se_method="hessian",
        inner_max_iter=500,
    )
    return model.fit(mce_irl_sample_df, state="state", action="action", id="id")


class TestMCEIRLInit:
    """Tests for MCEIRL initialization."""

    def test_mce_irl_init_defaults(self):
        """MCEIRL can be initialized with default parameters."""
        estimator = MCEIRL()

        assert estimator.n_states == 90
        assert estimator.n_actions == 2
        assert estimator.discount == 0.99
        assert estimator.se_method == "bootstrap"
        assert estimator.verbose is False
        assert estimator.feature_matrix is None
        assert estimator.feature_names is None

    def test_mce_irl_init_custom(self):
        """MCEIRL can be initialized with custom parameters."""
        features = np.random.randn(50, 3)
        estimator = MCEIRL(
            n_states=50,
            n_actions=3,
            discount=0.95,
            se_method="hessian",
            verbose=True,
            feature_matrix=features,
            feature_names=["f1", "f2", "f3"],
        )

        assert estimator.n_states == 50
        assert estimator.n_actions == 3
        assert estimator.discount == 0.95
        assert estimator.se_method == "hessian"
        assert estimator.verbose is True
        assert estimator.feature_matrix is not None
        assert estimator.feature_names == ["f1", "f2", "f3"]

    def test_mce_irl_init_with_feature_matrix(self):
        """MCEIRL can be initialized with a custom feature matrix."""
        n_states = 100
        n_features = 5
        features = np.random.randn(n_states, n_features)

        estimator = MCEIRL(
            n_states=n_states,
            feature_matrix=features,
        )

        assert estimator.feature_matrix is not None
        assert estimator.feature_matrix.shape == (n_states, n_features)


class TestMCEIRLFit:
    """Tests for MCEIRL.fit() method."""

    def test_mce_irl_fit_returns_self(self, mce_irl_sample_df):
        """fit() should return self for method chaining."""
        estimator = MCEIRL(
            n_states=20,
            discount=0.95,
            feature_matrix=_default_feature_matrix(20),
            feature_names=["f0"],
            verbose=False,
            se_method="hessian",  # Faster than bootstrap for testing
            inner_max_iter=500,
        )
        result = estimator.fit(
            data=mce_irl_sample_df,
            state="state",
            action="action",
            id="id",
        )

        assert result is estimator

    def test_mce_irl_fit_with_explicit_transitions(self, mce_irl_sample_df):
        """Can provide pre-estimated transitions to fit()."""
        # Create a simple transition matrix
        n_states = 20
        transitions = np.zeros((n_states, n_states))
        for s in range(n_states):
            for delta, p in [(0, 0.3), (1, 0.6), (2, 0.1)]:
                s_next = min(s + delta, n_states - 1)
                transitions[s, s_next] += p

        estimator = MCEIRL(
            n_states=n_states,
            discount=0.95,
            feature_matrix=_default_feature_matrix(n_states),
            feature_names=["f0"],
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        result = estimator.fit(
            data=mce_irl_sample_df,
            state="state",
            action="action",
            id="id",
            transitions=transitions,
        )

        assert result is estimator
        # Transitions should match what we provided
        np.testing.assert_allclose(estimator.transitions_, transitions, atol=1e-6)

    def test_mce_irl_fit_with_custom_features(self, mce_irl_sample_df):
        """Can fit with custom feature matrix."""
        n_states = 20
        n_features = 2

        # Create simple features: linear and indicator
        features = np.column_stack([
            np.arange(n_states) / n_states,
            (np.arange(n_states) > 10).astype(float),
        ])

        estimator = MCEIRL(
            n_states=n_states,
            discount=0.95,
            feature_matrix=features,
            feature_names=["linear", "high_state"],
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        result = estimator.fit(
            data=mce_irl_sample_df,
            state="state",
            action="action",
            id="id",
        )

        assert result is estimator
        assert len(estimator.params_) == n_features

    def test_mce_irl_fit_without_reward_spec_raises_for_multi_action(self, mce_irl_sample_df):
        estimator = MCEIRL(
            n_states=20,
            n_actions=2,
            discount=0.95,
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )

        with pytest.raises(ValueError, match="explicit reward specification"):
            estimator.fit(
                data=mce_irl_sample_df,
                state="state",
                action="action",
                id="id",
            )


class TestMCEIRLAttributes:
    """Tests for MCEIRL attributes after fit()."""

    def test_mce_irl_params_(self, mce_irl_fitted_estimator):
        """params_ should be a dict with feature names."""
        assert hasattr(mce_irl_fitted_estimator, "params_")
        assert isinstance(mce_irl_fitted_estimator.params_, dict)
        assert "f0" in mce_irl_fitted_estimator.params_

        # Parameters should be finite
        for name, val in mce_irl_fitted_estimator.params_.items():
            assert np.isfinite(val), f"Parameter {name} is not finite: {val}"

    def test_mce_irl_se_(self, mce_irl_fitted_estimator):
        """se_ should be a dict with standard errors."""
        assert hasattr(mce_irl_fitted_estimator, "se_")
        assert isinstance(mce_irl_fitted_estimator.se_, dict)
        assert "f0" in mce_irl_fitted_estimator.se_

        # Standard errors should be non-negative
        for name, val in mce_irl_fitted_estimator.se_.items():
            assert val >= 0, f"Standard error for {name} is negative: {val}"

    def test_mce_irl_coef_(self, mce_irl_fitted_estimator):
        """coef_ should be a numpy array of coefficients."""
        assert hasattr(mce_irl_fitted_estimator, "coef_")
        assert isinstance(mce_irl_fitted_estimator.coef_, np.ndarray)
        assert len(mce_irl_fitted_estimator.coef_) == 1  # Single feature in default

        # Should match params_
        assert np.isclose(mce_irl_fitted_estimator.coef_[0], mce_irl_fitted_estimator.params_["f0"])

    def test_mce_irl_reward_(self, mce_irl_fitted_estimator):
        """reward_ should be a numpy array with R(s) for each state."""
        assert hasattr(mce_irl_fitted_estimator, "reward_")
        assert isinstance(mce_irl_fitted_estimator.reward_, np.ndarray)
        assert len(mce_irl_fitted_estimator.reward_) == mce_irl_fitted_estimator.n_states

        # Reward should be finite
        assert np.all(np.isfinite(mce_irl_fitted_estimator.reward_))

    def test_mce_irl_policy_(self, mce_irl_fitted_estimator):
        """policy_ should be valid probability matrix."""
        assert hasattr(mce_irl_fitted_estimator, "policy_")
        assert isinstance(mce_irl_fitted_estimator.policy_, np.ndarray)
        assert mce_irl_fitted_estimator.policy_.shape == (
            mce_irl_fitted_estimator.n_states,
            mce_irl_fitted_estimator.n_actions,
        )

        # Rows should sum to 1
        row_sums = mce_irl_fitted_estimator.policy_.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(mce_irl_fitted_estimator.n_states),
            atol=1e-6
        )

        # All probabilities should be valid
        assert (mce_irl_fitted_estimator.policy_ >= 0).all()
        assert (mce_irl_fitted_estimator.policy_ <= 1).all()

    def test_mce_irl_log_likelihood_(self, mce_irl_fitted_estimator):
        """log_likelihood_ should be available and negative."""
        assert hasattr(mce_irl_fitted_estimator, "log_likelihood_")
        assert isinstance(mce_irl_fitted_estimator.log_likelihood_, float)
        assert mce_irl_fitted_estimator.log_likelihood_ < 0

    def test_mce_irl_value_function_(self, mce_irl_fitted_estimator):
        """value_function_ should be a numpy array."""
        assert hasattr(mce_irl_fitted_estimator, "value_function_")
        assert isinstance(mce_irl_fitted_estimator.value_function_, np.ndarray)
        assert len(mce_irl_fitted_estimator.value_function_) == mce_irl_fitted_estimator.n_states

    def test_mce_irl_transitions_(self, mce_irl_fitted_estimator):
        """transitions_ should be available after fit."""
        assert hasattr(mce_irl_fitted_estimator, "transitions_")
        assert isinstance(mce_irl_fitted_estimator.transitions_, np.ndarray)
        assert mce_irl_fitted_estimator.transitions_.shape == (
            mce_irl_fitted_estimator.n_states,
            mce_irl_fitted_estimator.n_states
        )

        # Rows should sum to 1
        row_sums = mce_irl_fitted_estimator.transitions_.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(mce_irl_fitted_estimator.n_states),
            atol=1e-6
        )

    def test_mce_irl_converged_(self, mce_irl_fitted_estimator):
        """converged_ should be a boolean."""
        assert hasattr(mce_irl_fitted_estimator, "converged_")
        assert isinstance(mce_irl_fitted_estimator.converged_, bool)


class TestMCEIRLSummary:
    """Tests for MCEIRL.summary() method."""

    def test_mce_irl_summary_returns_string(self, mce_irl_fitted_estimator):
        """summary() should return a formatted string."""
        summary = mce_irl_fitted_estimator.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_mce_irl_summary_contains_method(self, mce_irl_fitted_estimator):
        """summary() should mention MCE IRL method."""
        summary = mce_irl_fitted_estimator.summary()

        assert "MCE IRL" in summary or "Maximum Causal Entropy" in summary

    def test_mce_irl_summary_not_fitted(self):
        """summary() should indicate when not fitted."""
        estimator = MCEIRL()
        summary = estimator.summary()

        assert "Not fitted" in summary or "not fitted" in summary.lower()


class TestMCEIRLPredictProba:
    """Tests for MCEIRL.predict_proba() method."""

    def test_mce_irl_predict_proba_single_state(self, mce_irl_fitted_estimator):
        """predict_proba() works with a single state."""
        proba = mce_irl_fitted_estimator.predict_proba(states=np.array([0]))

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)  # 1 state, 2 actions

        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), [1.0], atol=1e-6)

        # Probabilities should be non-negative
        assert (proba >= 0).all()

    def test_mce_irl_predict_proba_multiple_states(self, mce_irl_fitted_estimator):
        """predict_proba() works with multiple states."""
        states = np.array([0, 5, 10, 15, 19])
        proba = mce_irl_fitted_estimator.predict_proba(states=states)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (5, 2)  # 5 states, 2 actions

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)

        # All probabilities should be non-negative
        assert (proba >= 0).all()

    def test_mce_irl_predict_proba_valid_probabilities(self, mce_irl_fitted_estimator):
        """predict_proba() should return valid probabilities."""
        states = np.array([0, 10, 15, 19])
        proba = mce_irl_fitted_estimator.predict_proba(states=states)

        # All probabilities should be valid
        assert (proba >= 0).all()
        assert (proba <= 1).all()

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-6)


class TestMCEIRLNotFitted:
    """Tests for MCEIRL error handling when not fitted."""

    def test_mce_irl_predict_proba_not_fitted(self):
        """predict_proba() should raise error when not fitted."""
        estimator = MCEIRL()

        with pytest.raises(RuntimeError, match="not fitted"):
            estimator.predict_proba(states=np.array([0, 1, 2]))

    def test_mce_irl_attributes_none_before_fit(self):
        """Fitted attributes should be None before fit()."""
        estimator = MCEIRL()

        assert estimator.params_ is None
        assert estimator.se_ is None
        assert estimator.coef_ is None
        assert estimator.reward_ is None
        assert estimator.policy_ is None
        assert estimator.log_likelihood_ is None
        assert estimator.value_function_ is None
        assert estimator.converged_ is None


class TestMCEIRLImport:
    """Tests for MCEIRL import structure."""

    def test_can_import_from_mce_irl_module(self):
        """MCEIRL can be imported from econirl.estimators.mce_irl."""
        # MCEIRL already imported at module level, verify it's accessible
        assert MCEIRL is not None

    def test_can_import_from_estimators(self):
        """MCEIRL can be imported from econirl.estimators."""
        from econirl.estimators import MCEIRL as EstimatorMCEIRL

        assert EstimatorMCEIRL is not None

    def test_mce_irl_in_all(self):
        """MCEIRL is in __all__ of econirl.estimators."""
        from econirl import estimators

        assert hasattr(estimators, "__all__")
        assert "MCEIRL" in estimators.__all__


class TestMCEIRLRepr:
    """Tests for MCEIRL __repr__ method."""

    def test_repr_not_fitted(self):
        """__repr__ should indicate not fitted."""
        estimator = MCEIRL(n_states=100, n_actions=4, discount=0.95)
        repr_str = repr(estimator)

        assert "MCEIRL" in repr_str
        assert "n_states=100" in repr_str
        assert "n_actions=4" in repr_str
        assert "discount=0.95" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self, mce_irl_fitted_estimator):
        """__repr__ should indicate fitted after fit()."""
        repr_str = repr(mce_irl_fitted_estimator)
        assert "fitted=True" in repr_str


class TestMCEIRLWithFeatures:
    """Tests for MCEIRL with custom feature matrices."""

    def test_params_after_fit_with_features(self, mce_irl_sample_df):
        """params_ should contain named features after fit."""
        n_states = 20
        features = np.arange(n_states).reshape(-1, 1) / 20
        model = MCEIRL(
            n_states=n_states,
            discount=0.95,
            feature_matrix=features,
            feature_names=["cost"],
            se_method="hessian",
            verbose=False,
            inner_max_iter=500,
        )
        model.fit(mce_irl_sample_df, state="state", action="action", id="id")

        assert model.params_ is not None
        assert "cost" in model.params_
        assert model.coef_ is not None

    def test_params_with_multiple_features(self, mce_irl_sample_df):
        """params_ should work with multiple features."""
        n_states = 20
        s = np.arange(n_states)
        features = np.column_stack([s / n_states, (s / n_states) ** 2])

        model = MCEIRL(
            n_states=n_states,
            discount=0.95,
            feature_matrix=features,
            feature_names=["linear", "quadratic"],
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        model.fit(mce_irl_sample_df, state="state", action="action", id="id")

        assert model.params_ is not None
        assert "linear" in model.params_
        assert "quadratic" in model.params_
        assert len(model.coef_) == 2


class TestMCEIRLMethodChaining:
    """Tests for MCEIRL method chaining."""

    def test_fit_and_predict_chain(self, mce_irl_sample_df):
        """fit() followed by predict_proba() should work."""
        model = MCEIRL(
            n_states=20,
            discount=0.95,
            feature_matrix=_default_feature_matrix(20),
            feature_names=["f0"],
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        proba = model.fit(
            mce_irl_sample_df, state="state", action="action", id="id"
        ).predict_proba(np.array([0, 5, 10]))

        assert proba.shape == (3, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)
