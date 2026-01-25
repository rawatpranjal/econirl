"""Tests for sklearn-style MCEIRL estimator.

Tests the MCEIRL estimator class which provides a scikit-learn style
interface for Maximum Causal Entropy Inverse Reinforcement Learning (Ziebart 2010).
"""

import pytest
import numpy as np
import pandas as pd
import torch

from econirl.core.types import Panel, Trajectory


class TestMCEIRLInit:
    """Tests for MCEIRL initialization."""

    def test_mce_irl_init_defaults(self):
        """MCEIRL can be initialized with default parameters."""
        from econirl.estimators.mce_irl import MCEIRL

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
        from econirl.estimators.mce_irl import MCEIRL

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
        from econirl.estimators.mce_irl import MCEIRL

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

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n_individuals = 10
        n_periods = 20
        n_states = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                # Simple stochastic policy based on state
                action = 1 if state > 10 or np.random.random() < 0.1 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1]),
                    n_states - 1
                )
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                })
                state = next_state

        return pd.DataFrame(data)

    def test_mce_irl_fit_returns_self(self, sample_dataframe):
        """fit() should return self for method chaining."""
        from econirl.estimators.mce_irl import MCEIRL

        estimator = MCEIRL(
            n_states=20,
            discount=0.95,
            verbose=False,
            se_method="hessian",  # Faster than bootstrap for testing
            inner_max_iter=500,
        )
        result = estimator.fit(
            data=sample_dataframe,
            state="state",
            action="action",
            id="id",
        )

        assert result is estimator

    def test_mce_irl_fit_with_explicit_transitions(self, sample_dataframe):
        """Can provide pre-estimated transitions to fit()."""
        from econirl.estimators.mce_irl import MCEIRL

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
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        result = estimator.fit(
            data=sample_dataframe,
            state="state",
            action="action",
            id="id",
            transitions=transitions,
        )

        assert result is estimator
        # Transitions should match what we provided
        np.testing.assert_allclose(estimator.transitions_, transitions, atol=1e-6)

    def test_mce_irl_fit_with_custom_features(self, sample_dataframe):
        """Can fit with custom feature matrix."""
        from econirl.estimators.mce_irl import MCEIRL

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
            data=sample_dataframe,
            state="state",
            action="action",
            id="id",
        )

        assert result is estimator
        assert len(estimator.params_) == n_features


class TestMCEIRLAttributes:
    """Tests for MCEIRL attributes after fit()."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a MCEIRL estimator."""
        from econirl.estimators.mce_irl import MCEIRL

        np.random.seed(42)
        n_individuals = 15
        n_periods = 25
        n_states = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 15 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                    n_states - 1
                )
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        # Use simple single feature (state index)
        estimator = MCEIRL(
            n_states=n_states,
            discount=0.95,
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        estimator.fit(
            data=df,
            state="state",
            action="action",
            id="id",
        )

        return estimator

    def test_mce_irl_params_(self, fitted_estimator):
        """params_ should be a dict with feature names."""
        assert hasattr(fitted_estimator, "params_")
        assert isinstance(fitted_estimator.params_, dict)
        assert "f0" in fitted_estimator.params_

        # Parameters should be finite
        for name, val in fitted_estimator.params_.items():
            assert np.isfinite(val), f"Parameter {name} is not finite: {val}"

    def test_mce_irl_se_(self, fitted_estimator):
        """se_ should be a dict with standard errors."""
        assert hasattr(fitted_estimator, "se_")
        assert isinstance(fitted_estimator.se_, dict)
        assert "f0" in fitted_estimator.se_

        # Standard errors should be non-negative
        for name, val in fitted_estimator.se_.items():
            assert val >= 0, f"Standard error for {name} is negative: {val}"

    def test_mce_irl_coef_(self, fitted_estimator):
        """coef_ should be a numpy array of coefficients."""
        assert hasattr(fitted_estimator, "coef_")
        assert isinstance(fitted_estimator.coef_, np.ndarray)
        assert len(fitted_estimator.coef_) == 1  # Single feature in default

        # Should match params_
        assert np.isclose(fitted_estimator.coef_[0], fitted_estimator.params_["f0"])

    def test_mce_irl_reward_(self, fitted_estimator):
        """reward_ should be a numpy array with R(s) for each state."""
        assert hasattr(fitted_estimator, "reward_")
        assert isinstance(fitted_estimator.reward_, np.ndarray)
        assert len(fitted_estimator.reward_) == fitted_estimator.n_states

        # Reward should be finite
        assert np.all(np.isfinite(fitted_estimator.reward_))

    def test_mce_irl_policy_(self, fitted_estimator):
        """policy_ should be valid probability matrix."""
        assert hasattr(fitted_estimator, "policy_")
        assert isinstance(fitted_estimator.policy_, np.ndarray)
        assert fitted_estimator.policy_.shape == (
            fitted_estimator.n_states,
            fitted_estimator.n_actions,
        )

        # Rows should sum to 1
        row_sums = fitted_estimator.policy_.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(fitted_estimator.n_states),
            atol=1e-6
        )

        # All probabilities should be valid
        assert (fitted_estimator.policy_ >= 0).all()
        assert (fitted_estimator.policy_ <= 1).all()

    def test_mce_irl_log_likelihood_(self, fitted_estimator):
        """log_likelihood_ should be available and negative."""
        assert hasattr(fitted_estimator, "log_likelihood_")
        assert isinstance(fitted_estimator.log_likelihood_, float)
        assert fitted_estimator.log_likelihood_ < 0

    def test_mce_irl_value_function_(self, fitted_estimator):
        """value_function_ should be a numpy array."""
        assert hasattr(fitted_estimator, "value_function_")
        assert isinstance(fitted_estimator.value_function_, np.ndarray)
        assert len(fitted_estimator.value_function_) == fitted_estimator.n_states

    def test_mce_irl_transitions_(self, fitted_estimator):
        """transitions_ should be available after fit."""
        assert hasattr(fitted_estimator, "transitions_")
        assert isinstance(fitted_estimator.transitions_, np.ndarray)
        assert fitted_estimator.transitions_.shape == (
            fitted_estimator.n_states,
            fitted_estimator.n_states
        )

        # Rows should sum to 1
        row_sums = fitted_estimator.transitions_.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(fitted_estimator.n_states),
            atol=1e-6
        )

    def test_mce_irl_converged_(self, fitted_estimator):
        """converged_ should be a boolean."""
        assert hasattr(fitted_estimator, "converged_")
        assert isinstance(fitted_estimator.converged_, bool)


class TestMCEIRLSummary:
    """Tests for MCEIRL.summary() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a MCEIRL estimator."""
        from econirl.estimators.mce_irl import MCEIRL

        np.random.seed(42)
        n_individuals = 10
        n_periods = 20
        n_states = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 15 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.4, 0.55, 0.05]),
                    n_states - 1
                )
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = MCEIRL(
            n_states=n_states,
            discount=0.95,
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        estimator.fit(
            data=df,
            state="state",
            action="action",
            id="id",
        )

        return estimator

    def test_mce_irl_summary_returns_string(self, fitted_estimator):
        """summary() should return a formatted string."""
        summary = fitted_estimator.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_mce_irl_summary_contains_method(self, fitted_estimator):
        """summary() should mention MCE IRL method."""
        summary = fitted_estimator.summary()

        assert "MCE IRL" in summary or "Maximum Causal Entropy" in summary

    def test_mce_irl_summary_not_fitted(self):
        """summary() should indicate when not fitted."""
        from econirl.estimators.mce_irl import MCEIRL

        estimator = MCEIRL()
        summary = estimator.summary()

        assert "Not fitted" in summary or "not fitted" in summary.lower()


class TestMCEIRLPredictProba:
    """Tests for MCEIRL.predict_proba() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a MCEIRL estimator."""
        from econirl.estimators.mce_irl import MCEIRL

        np.random.seed(42)
        n_individuals = 15
        n_periods = 25
        n_states = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 15 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                    n_states - 1
                )
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = MCEIRL(
            n_states=n_states,
            discount=0.95,
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        estimator.fit(
            data=df,
            state="state",
            action="action",
            id="id",
        )

        return estimator

    def test_mce_irl_predict_proba_single_state(self, fitted_estimator):
        """predict_proba() works with a single state."""
        proba = fitted_estimator.predict_proba(states=np.array([0]))

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)  # 1 state, 2 actions

        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), [1.0], atol=1e-6)

        # Probabilities should be non-negative
        assert (proba >= 0).all()

    def test_mce_irl_predict_proba_multiple_states(self, fitted_estimator):
        """predict_proba() works with multiple states."""
        states = np.array([0, 5, 10, 15, 19])
        proba = fitted_estimator.predict_proba(states=states)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (5, 2)  # 5 states, 2 actions

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)

        # All probabilities should be non-negative
        assert (proba >= 0).all()

    def test_mce_irl_predict_proba_valid_probabilities(self, fitted_estimator):
        """predict_proba() should return valid probabilities."""
        states = np.array([0, 10, 15, 19])
        proba = fitted_estimator.predict_proba(states=states)

        # All probabilities should be valid
        assert (proba >= 0).all()
        assert (proba <= 1).all()

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-6)


class TestMCEIRLNotFitted:
    """Tests for MCEIRL error handling when not fitted."""

    def test_mce_irl_predict_proba_not_fitted(self):
        """predict_proba() should raise error when not fitted."""
        from econirl.estimators.mce_irl import MCEIRL

        estimator = MCEIRL()

        with pytest.raises(RuntimeError, match="not fitted"):
            estimator.predict_proba(states=np.array([0, 1, 2]))

    def test_mce_irl_attributes_none_before_fit(self):
        """Fitted attributes should be None before fit()."""
        from econirl.estimators.mce_irl import MCEIRL

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
        from econirl.estimators.mce_irl import MCEIRL

        assert MCEIRL is not None

    def test_can_import_from_estimators(self):
        """MCEIRL can be imported from econirl.estimators."""
        from econirl.estimators import MCEIRL

        assert MCEIRL is not None

    def test_mce_irl_in_all(self):
        """MCEIRL is in __all__ of econirl.estimators."""
        from econirl import estimators

        assert hasattr(estimators, "__all__")
        assert "MCEIRL" in estimators.__all__


class TestMCEIRLRepr:
    """Tests for MCEIRL __repr__ method."""

    def test_repr_not_fitted(self):
        """__repr__ should indicate not fitted."""
        from econirl.estimators.mce_irl import MCEIRL

        estimator = MCEIRL(n_states=100, n_actions=4, discount=0.95)
        repr_str = repr(estimator)

        assert "MCEIRL" in repr_str
        assert "n_states=100" in repr_str
        assert "n_actions=4" in repr_str
        assert "discount=0.95" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self):
        """__repr__ should indicate fitted after fit()."""
        from econirl.estimators.mce_irl import MCEIRL

        np.random.seed(42)
        n_states = 20
        data = []
        for i in range(5):
            state = 0
            for t in range(10):
                action = 1 if state > 15 else 0
                next_state = 0 if action == 1 else min(state + 1, n_states - 1)
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = MCEIRL(
            n_states=n_states,
            discount=0.95,
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        estimator.fit(df, state="state", action="action", id="id")

        repr_str = repr(estimator)
        assert "fitted=True" in repr_str


class TestMCEIRLWithFeatures:
    """Tests for MCEIRL with custom feature matrices."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n_individuals = 10
        n_periods = 20
        n_states = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 10 or np.random.random() < 0.1 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2]),
                    n_states - 1
                )
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        return pd.DataFrame(data)

    def test_params_after_fit_with_features(self, sample_df):
        """params_ should contain named features after fit."""
        from econirl.estimators.mce_irl import MCEIRL

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
        model.fit(sample_df, state="state", action="action", id="id")

        assert model.params_ is not None
        assert "cost" in model.params_
        assert model.coef_ is not None

    def test_params_with_multiple_features(self, sample_df):
        """params_ should work with multiple features."""
        from econirl.estimators.mce_irl import MCEIRL

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
        model.fit(sample_df, state="state", action="action", id="id")

        assert model.params_ is not None
        assert "linear" in model.params_
        assert "quadratic" in model.params_
        assert len(model.coef_) == 2


class TestMCEIRLMethodChaining:
    """Tests for MCEIRL method chaining."""

    def test_fit_and_predict_chain(self):
        """fit() followed by predict_proba() should work."""
        from econirl.estimators.mce_irl import MCEIRL

        np.random.seed(42)
        n_states = 20
        data = []
        for i in range(5):
            state = 0
            for t in range(15):
                action = 1 if state > 15 else 0
                next_state = 0 if action == 1 else min(state + 1, n_states - 1)
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)

        model = MCEIRL(
            n_states=n_states,
            discount=0.95,
            verbose=False,
            se_method="hessian",
            inner_max_iter=500,
        )
        proba = model.fit(
            df, state="state", action="action", id="id"
        ).predict_proba(np.array([0, 5, 10]))

        assert proba.shape == (3, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)
