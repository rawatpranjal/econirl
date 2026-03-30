"""Tests for sklearn-style MaxMarginIRL estimator.

Tests the MaxMarginIRL estimator class which provides a scikit-learn style
interface for the Maximum Margin IRL algorithm (Abbeel & Ng 2004).
"""

import pytest
import numpy as np
import pandas as pd
import torch


class TestMaxMarginIRLInit:
    """Tests for MaxMarginIRL initialization."""

    def test_init_defaults(self):
        """MaxMarginIRL can be initialized with default parameters."""
        from econirl.estimators import MaxMarginIRL

        estimator = MaxMarginIRL()

        assert estimator.n_states == 90
        assert estimator.n_actions == 2
        assert estimator.discount == 0.99
        assert estimator.n_features == 90  # defaults to n_states
        assert estimator.features is None
        assert estimator.feature_names is None
        assert estimator.max_iterations == 50
        assert estimator.margin_tol == 1e-4
        assert estimator.se_method == "asymptotic"
        assert estimator.verbose is False

    def test_init_custom(self):
        """MaxMarginIRL can be initialized with custom parameters."""
        from econirl.estimators import MaxMarginIRL

        features = np.random.randn(50, 5)
        feature_names = ["f0", "f1", "f2", "f3", "f4"]

        estimator = MaxMarginIRL(
            n_states=50,
            n_actions=3,
            discount=0.95,
            n_features=5,
            features=features,
            feature_names=feature_names,
            max_iterations=100,
            margin_tol=1e-5,
            se_method="asymptotic",
            verbose=True,
        )

        assert estimator.n_states == 50
        assert estimator.n_actions == 3
        assert estimator.discount == 0.95
        assert estimator.n_features == 5
        assert estimator.features is not None
        assert estimator.feature_names == feature_names
        assert estimator.max_iterations == 100
        assert estimator.margin_tol == 1e-5
        assert estimator.verbose is True

    def test_init_attributes_none_before_fit(self):
        """Fitted attributes should be None before fit()."""
        from econirl.estimators import MaxMarginIRL

        estimator = MaxMarginIRL()

        assert estimator.params_ is None
        assert estimator.se_ is None
        assert estimator.coef_ is None
        assert estimator.reward_ is None
        assert estimator.margin_ is None
        assert estimator.value_function_ is None
        assert estimator.transitions_ is None
        assert estimator.converged_ is None


class TestMaxMarginIRLFit:
    """Tests for MaxMarginIRL.fit() method."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n_individuals = 10
        n_periods = 20
        n_states = 30  # Smaller state space for faster tests

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                # Simple stochastic policy
                action = 1 if state > 20 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1]),
                    n_states - 1,
                )
                data.append({
                    "agent_id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                })
                state = next_state

        return pd.DataFrame(data)

    def test_fit_returns_self(self, sample_dataframe):
        """fit() should return self for method chaining."""
        from econirl.estimators import MaxMarginIRL

        estimator = MaxMarginIRL(n_states=30, n_features=5, verbose=False)
        result = estimator.fit(
            data=sample_dataframe,
            state="state",
            action="action",
            id="agent_id",
        )

        assert result is estimator

    def test_fit_with_explicit_transitions(self, sample_dataframe):
        """Can provide pre-estimated transitions to fit()."""
        from econirl.estimators import MaxMarginIRL

        n_states = 30
        transitions = np.zeros((n_states, n_states))
        for s in range(n_states):
            for delta, p in [(0, 0.3), (1, 0.6), (2, 0.1)]:
                s_next = min(s + delta, n_states - 1)
                transitions[s, s_next] += p

        estimator = MaxMarginIRL(n_states=n_states, n_features=5, verbose=False)
        result = estimator.fit(
            data=sample_dataframe,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )

        assert result is estimator
        np.testing.assert_allclose(estimator.transitions_, transitions, atol=1e-6)

    def test_fit_with_custom_features(self, sample_dataframe):
        """Can provide custom feature matrix to fit()."""
        from econirl.estimators import MaxMarginIRL

        n_states = 30
        n_features = 3
        features = np.random.randn(n_states, n_features)
        feature_names = ["feature_a", "feature_b", "feature_c"]

        estimator = MaxMarginIRL(
            n_states=n_states,
            features=features,
            feature_names=feature_names,
            verbose=False,
        )
        estimator.fit(
            data=sample_dataframe,
            state="state",
            action="action",
            id="agent_id",
        )

        # Parameter names should match feature names
        assert set(estimator.params_.keys()) == set(feature_names)


class TestMaxMarginIRLAttributes:
    """Tests for MaxMarginIRL attributes after fit()."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a MaxMarginIRL estimator."""
        from econirl.estimators import MaxMarginIRL

        np.random.seed(42)
        n_individuals = 15
        n_periods = 25
        n_states = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 20 or np.random.random() < 0.03 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                    n_states - 1,
                )
                data.append({
                    "agent_id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = MaxMarginIRL(n_states=n_states, n_features=5, verbose=False)
        estimator.fit(
            data=df,
            state="state",
            action="action",
            id="agent_id",
        )

        return estimator

    def test_params_(self, fitted_estimator):
        """params_ should be a dict with feature names."""
        assert hasattr(fitted_estimator, "params_")
        assert isinstance(fitted_estimator.params_, dict)
        assert len(fitted_estimator.params_) == 5  # n_features

        # All parameters should be finite
        for name, val in fitted_estimator.params_.items():
            assert np.isfinite(val), f"Parameter {name} is not finite: {val}"

    def test_se_(self, fitted_estimator):
        """se_ should be a dict with standard errors."""
        assert hasattr(fitted_estimator, "se_")
        assert isinstance(fitted_estimator.se_, dict)
        assert len(fitted_estimator.se_) == 5  # n_features

        # Standard errors should be non-negative or NaN
        # (NaN is acceptable for Max Margin IRL since it uses a pseudo-Hessian
        # which doesn't have a proper likelihood-based inference framework)
        for name, val in fitted_estimator.se_.items():
            assert val >= 0 or np.isnan(val), f"SE for {name} is invalid: {val}"

    def test_coef_(self, fitted_estimator):
        """coef_ should be a numpy array of coefficients."""
        assert hasattr(fitted_estimator, "coef_")
        assert isinstance(fitted_estimator.coef_, np.ndarray)
        assert len(fitted_estimator.coef_) == 5  # n_features

        # coef_ should match params_ values
        for i, name in enumerate(fitted_estimator.params_.keys()):
            assert np.isclose(
                fitted_estimator.coef_[i],
                fitted_estimator.params_[name],
            )

    def test_reward_(self, fitted_estimator):
        """reward_ should be a numpy array of rewards for each state."""
        assert hasattr(fitted_estimator, "reward_")
        assert isinstance(fitted_estimator.reward_, np.ndarray)
        assert len(fitted_estimator.reward_) == fitted_estimator.n_states

        # Rewards should be finite
        assert np.all(np.isfinite(fitted_estimator.reward_))

    def test_margin_(self, fitted_estimator):
        """margin_ should be available after fit."""
        assert hasattr(fitted_estimator, "margin_")
        # Margin may be None if not stored in metadata, but if present should be a number
        if fitted_estimator.margin_ is not None:
            assert isinstance(fitted_estimator.margin_, float)

    def test_value_function_(self, fitted_estimator):
        """value_function_ should be a numpy array."""
        assert hasattr(fitted_estimator, "value_function_")
        assert isinstance(fitted_estimator.value_function_, np.ndarray)
        assert len(fitted_estimator.value_function_) == fitted_estimator.n_states

    def test_transitions_(self, fitted_estimator):
        """transitions_ should be available after fit."""
        assert hasattr(fitted_estimator, "transitions_")
        assert isinstance(fitted_estimator.transitions_, np.ndarray)
        assert fitted_estimator.transitions_.shape == (
            fitted_estimator.n_states,
            fitted_estimator.n_states,
        )

        # Rows should sum to 1
        row_sums = fitted_estimator.transitions_.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(fitted_estimator.n_states),
            atol=1e-6,
        )

    def test_converged_(self, fitted_estimator):
        """converged_ should be a boolean."""
        assert hasattr(fitted_estimator, "converged_")
        assert isinstance(fitted_estimator.converged_, bool)


class TestMaxMarginIRLSummary:
    """Tests for MaxMarginIRL.summary() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a MaxMarginIRL estimator."""
        from econirl.estimators import MaxMarginIRL

        np.random.seed(42)
        n_individuals = 10
        n_periods = 20
        n_states = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 20 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.4, 0.55, 0.05]),
                    n_states - 1,
                )
                data.append({
                    "agent_id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = MaxMarginIRL(n_states=n_states, n_features=5, verbose=False)
        estimator.fit(
            data=df,
            state="state",
            action="action",
            id="agent_id",
        )

        return estimator

    def test_summary_returns_string(self, fitted_estimator):
        """summary() should return a formatted string."""
        summary = fitted_estimator.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_not_fitted(self):
        """summary() should indicate not fitted if model hasn't been fit."""
        from econirl.estimators import MaxMarginIRL

        estimator = MaxMarginIRL()
        summary = estimator.summary()

        assert "Not fitted" in summary or "fit()" in summary


class TestMaxMarginIRLPredictProba:
    """Tests for MaxMarginIRL.predict_proba() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit a MaxMarginIRL estimator."""
        from econirl.estimators import MaxMarginIRL

        np.random.seed(42)
        n_individuals = 15
        n_periods = 25
        n_states = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 20 or np.random.random() < 0.03 else 0
                next_state = 0 if action == 1 else min(
                    state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                    n_states - 1,
                )
                data.append({
                    "agent_id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = MaxMarginIRL(n_states=n_states, n_features=5, verbose=False)
        estimator.fit(
            data=df,
            state="state",
            action="action",
            id="agent_id",
        )

        return estimator

    def test_predict_proba_single_state(self, fitted_estimator):
        """predict_proba() works with a single state."""
        proba = fitted_estimator.predict_proba(states=np.array([0]))

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)  # 1 state, 2 actions

        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), [1.0], atol=1e-6)

        # Probabilities should be non-negative
        assert (proba >= 0).all()

    def test_predict_proba_multiple_states(self, fitted_estimator):
        """predict_proba() works with multiple states."""
        states = np.array([0, 5, 10, 20, 29])
        proba = fitted_estimator.predict_proba(states=states)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (5, 2)  # 5 states, 2 actions

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)

        # All probabilities should be non-negative
        assert (proba >= 0).all()

    def test_predict_proba_valid_probabilities(self, fitted_estimator):
        """predict_proba() should return valid probabilities across all states."""
        states = np.array([0, 10, 20, 29])
        proba = fitted_estimator.predict_proba(states=states)

        # All probabilities should be valid
        assert (proba >= 0).all()
        assert (proba <= 1).all()

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-6)

    def test_predict_proba_not_fitted_raises(self):
        """predict_proba() should raise error if model not fitted."""
        from econirl.estimators import MaxMarginIRL

        estimator = MaxMarginIRL()

        with pytest.raises(RuntimeError):
            estimator.predict_proba(states=np.array([0]))


class TestMaxMarginIRLImport:
    """Tests for MaxMarginIRL import structure."""

    def test_can_import_from_estimators(self):
        """MaxMarginIRL can be imported from econirl.estimators."""
        from econirl.estimators import MaxMarginIRL

        assert MaxMarginIRL is not None

    def test_in_all(self):
        """MaxMarginIRL is in __all__ of econirl.estimators."""
        from econirl import estimators

        assert hasattr(estimators, "__all__")
        assert "MaxMarginIRL" in estimators.__all__


class TestMaxMarginIRLRepr:
    """Tests for MaxMarginIRL.__repr__() method."""

    def test_repr_not_fitted(self):
        """__repr__ should indicate not fitted."""
        from econirl.estimators import MaxMarginIRL

        estimator = MaxMarginIRL(n_states=50, n_actions=3, discount=0.95)
        repr_str = repr(estimator)

        assert "MaxMarginIRL" in repr_str
        assert "n_states=50" in repr_str
        assert "n_actions=3" in repr_str
        assert "discount=0.95" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self):
        """__repr__ should indicate fitted after fit()."""
        from econirl.estimators import MaxMarginIRL

        np.random.seed(42)
        n_states = 20
        data = []
        for i in range(5):
            state = 0
            for t in range(10):
                action = 1 if state > 15 else 0
                next_state = 0 if action == 1 else min(state + 1, n_states - 1)
                data.append({
                    "agent_id": i,
                    "state": state,
                    "action": action,
                })
                state = next_state

        df = pd.DataFrame(data)
        estimator = MaxMarginIRL(n_states=n_states, n_features=3, verbose=False)
        estimator.fit(df, state="state", action="action", id="agent_id")

        repr_str = repr(estimator)
        assert "fitted=True" in repr_str
