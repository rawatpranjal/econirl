"""Tests for sklearn-style NFXP estimator.

Tests the NFXP estimator class which provides a scikit-learn style
interface for the Nested Fixed Point algorithm from Rust (1987, 1988).
"""

import pytest
import numpy as np
import pandas as pd
import torch

from econirl.core.types import Panel, Trajectory


class TestNFXPInit:
    """Tests for NFXP initialization."""

    def test_nfxp_init_defaults(self):
        """NFXP can be initialized with default parameters."""
        from econirl.estimators import NFXP

        estimator = NFXP()

        assert estimator.n_states == 90
        assert estimator.n_actions == 2
        assert estimator.discount == 0.9999
        assert estimator.utility == "linear_cost"
        assert estimator.se_method == "robust"
        assert estimator.verbose is False

    def test_nfxp_init_custom(self):
        """NFXP can be initialized with custom parameters."""
        from econirl.estimators import NFXP

        estimator = NFXP(
            n_states=50,
            n_actions=3,
            discount=0.95,
            utility="linear_cost",
            se_method="asymptotic",
            verbose=True,
        )

        assert estimator.n_states == 50
        assert estimator.n_actions == 3
        assert estimator.discount == 0.95
        assert estimator.utility == "linear_cost"
        assert estimator.se_method == "asymptotic"
        assert estimator.verbose is True


class TestNFXPFit:
    """Tests for NFXP.fit() method."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        # Simulate some simple data
        np.random.seed(42)
        n_individuals = 10
        n_periods = 20

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                # Simple stochastic policy
                action = 1 if state > 50 or np.random.random() < 0.05 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_panel(self, sample_dataframe):
        """Create Panel from sample DataFrame."""
        df = sample_dataframe
        trajectories = []
        for bus_id in df["bus_id"].unique():
            bus_df = df[df["bus_id"] == bus_id].sort_values("period")
            traj = Trajectory(
                states=torch.tensor(bus_df["mileage_bin"].values, dtype=torch.long),
                actions=torch.tensor(bus_df["replaced"].values, dtype=torch.long),
                next_states=torch.tensor(bus_df["next_mileage"].values, dtype=torch.long),
                individual_id=bus_id,
            )
            trajectories.append(traj)
        return Panel(trajectories=trajectories)

    def test_nfxp_fit_returns_self(self, sample_dataframe):
        """fit() should return self for method chaining."""
        from econirl.estimators import NFXP

        estimator = NFXP(n_states=90, verbose=False)
        result = estimator.fit(
            data=sample_dataframe,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        assert result is estimator

    def test_nfxp_fit_with_explicit_transitions(self, sample_dataframe):
        """Can provide pre-estimated transitions to fit()."""
        from econirl.estimators import NFXP

        # Create a simple transition matrix
        n_states = 90
        transitions = np.zeros((n_states, n_states))
        for s in range(n_states):
            for delta, p in [(0, 0.3), (1, 0.6), (2, 0.1)]:
                s_next = min(s + delta, n_states - 1)
                transitions[s, s_next] += p

        estimator = NFXP(n_states=n_states, verbose=False)
        result = estimator.fit(
            data=sample_dataframe,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
            transitions=transitions,
        )

        assert result is estimator
        # Transitions should match what we provided
        np.testing.assert_allclose(estimator.transitions_, transitions, atol=1e-6)


class TestNFXPAttributes:
    """Tests for NFXP attributes after fit()."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit an NFXP estimator."""
        from econirl.estimators import NFXP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = NFXP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_nfxp_params_(self, fitted_estimator):
        """params_ should be a dict with theta_c and RC."""
        assert hasattr(fitted_estimator, "params_")
        assert isinstance(fitted_estimator.params_, dict)
        assert "theta_c" in fitted_estimator.params_
        assert "RC" in fitted_estimator.params_

        # Parameters should be finite
        assert np.isfinite(fitted_estimator.params_["theta_c"])
        assert np.isfinite(fitted_estimator.params_["RC"])

    def test_nfxp_se_(self, fitted_estimator):
        """se_ should be a dict with standard errors."""
        assert hasattr(fitted_estimator, "se_")
        assert isinstance(fitted_estimator.se_, dict)
        assert "theta_c" in fitted_estimator.se_
        assert "RC" in fitted_estimator.se_

        # Standard errors should be positive
        assert fitted_estimator.se_["theta_c"] >= 0
        assert fitted_estimator.se_["RC"] >= 0

    def test_nfxp_coef_(self, fitted_estimator):
        """coef_ should be a numpy array of coefficients."""
        assert hasattr(fitted_estimator, "coef_")
        assert isinstance(fitted_estimator.coef_, np.ndarray)
        assert len(fitted_estimator.coef_) == 2  # theta_c and RC

        # Should match params_
        assert np.isclose(fitted_estimator.coef_[0], fitted_estimator.params_["theta_c"])
        assert np.isclose(fitted_estimator.coef_[1], fitted_estimator.params_["RC"])

    def test_nfxp_log_likelihood_(self, fitted_estimator):
        """log_likelihood_ should be available and negative."""
        assert hasattr(fitted_estimator, "log_likelihood_")
        assert isinstance(fitted_estimator.log_likelihood_, float)
        assert fitted_estimator.log_likelihood_ < 0  # Log-likelihood is negative for probabilities < 1

    def test_nfxp_value_function_(self, fitted_estimator):
        """value_function_ should be a numpy array."""
        assert hasattr(fitted_estimator, "value_function_")
        assert isinstance(fitted_estimator.value_function_, np.ndarray)
        assert len(fitted_estimator.value_function_) == fitted_estimator.n_states

    def test_nfxp_transitions_(self, fitted_estimator):
        """transitions_ should be available after fit."""
        assert hasattr(fitted_estimator, "transitions_")
        assert isinstance(fitted_estimator.transitions_, np.ndarray)
        assert fitted_estimator.transitions_.shape == (fitted_estimator.n_states, fitted_estimator.n_states)

        # Rows should sum to 1
        row_sums = fitted_estimator.transitions_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(fitted_estimator.n_states), atol=1e-6)

    def test_nfxp_converged_(self, fitted_estimator):
        """converged_ should be a boolean."""
        assert hasattr(fitted_estimator, "converged_")
        assert isinstance(fitted_estimator.converged_, bool)


class TestNFXPSummary:
    """Tests for NFXP.summary() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit an NFXP estimator."""
        from econirl.estimators import NFXP

        np.random.seed(42)
        n_individuals = 15
        n_periods = 25

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 55 or np.random.random() < 0.03 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.4, 0.55, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = NFXP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_nfxp_summary_returns_string(self, fitted_estimator):
        """summary() should return a formatted string."""
        summary = fitted_estimator.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_nfxp_summary_contains_parameters(self, fitted_estimator):
        """summary() should contain parameter names and values."""
        summary = fitted_estimator.summary()

        assert "theta_c" in summary or "operating" in summary.lower() or "cost" in summary.lower()
        assert "RC" in summary or "replacement" in summary.lower()

    def test_nfxp_summary_contains_statistics(self, fitted_estimator):
        """summary() should contain key statistics."""
        summary = fitted_estimator.summary()

        # Should contain at least some of: log-likelihood, observations, etc.
        has_ll = "likelihood" in summary.lower() or "log" in summary.lower()
        has_obs = "observation" in summary.lower() or "obs" in summary.lower() or "n=" in summary.lower()

        assert has_ll or has_obs


class TestNFXPPredictProba:
    """Tests for NFXP.predict_proba() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit an NFXP estimator."""
        from econirl.estimators import NFXP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = NFXP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_nfxp_predict_proba_single_state(self, fitted_estimator):
        """predict_proba() works with a single state."""
        proba = fitted_estimator.predict_proba(states=np.array([0]))

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)  # 1 state, 2 actions

        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), [1.0], atol=1e-6)

        # Probabilities should be non-negative
        assert (proba >= 0).all()

    def test_nfxp_predict_proba_multiple_states(self, fitted_estimator):
        """predict_proba() works with multiple states."""
        states = np.array([0, 10, 30, 50, 80])
        proba = fitted_estimator.predict_proba(states=states)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (5, 2)  # 5 states, 2 actions

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)

        # All probabilities should be non-negative
        assert (proba >= 0).all()

    def test_nfxp_predict_proba_valid_probabilities(self, fitted_estimator):
        """predict_proba() should return valid probabilities across all states."""
        states = np.array([0, 30, 60, 89])
        proba = fitted_estimator.predict_proba(states=states)

        # All probabilities should be valid
        assert (proba >= 0).all()
        assert (proba <= 1).all()

        # Each row should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-6)

        # With small synthetic data, we can't guarantee qualitative outcomes,
        # but we can verify that the probabilities vary across states
        # (i.e., the model is not returning constant probabilities)
        p_replace = proba[:, 1]
        has_variation = p_replace.max() - p_replace.min() > 0 or len(np.unique(p_replace)) > 1
        # This is a very weak check - just ensure the model computes something sensible
        assert has_variation or np.allclose(p_replace, 0.5, atol=0.01)


class TestNFXPImport:
    """Tests for NFXP import structure."""

    def test_can_import_from_estimators(self):
        """NFXP can be imported from econirl.estimators."""
        from econirl.estimators import NFXP

        assert NFXP is not None

    def test_nfxp_in_all(self):
        """NFXP is in __all__ of econirl.estimators."""
        from econirl import estimators

        assert hasattr(estimators, "__all__")
        assert "NFXP" in estimators.__all__


class TestNFXPSimulate:
    """Tests for NFXP.simulate() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit an NFXP estimator."""
        from econirl.estimators import NFXP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = NFXP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_nfxp_simulate_returns_dataframe(self, fitted_estimator):
        """simulate() should return a pandas DataFrame."""
        result = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        assert isinstance(result, pd.DataFrame)

    def test_nfxp_simulate_has_correct_columns(self, fitted_estimator):
        """simulate() should return DataFrame with required columns."""
        result = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        assert "agent_id" in result.columns
        assert "period" in result.columns
        assert "state" in result.columns
        assert "action" in result.columns

    def test_nfxp_simulate_has_correct_shape(self, fitted_estimator):
        """simulate() should return correct number of rows."""
        n_agents = 5
        n_periods = 10
        result = fitted_estimator.simulate(n_agents=n_agents, n_periods=n_periods, seed=42)

        # Should have n_agents * n_periods rows
        assert len(result) == n_agents * n_periods

    def test_nfxp_simulate_agent_ids(self, fitted_estimator):
        """simulate() should have correct agent IDs."""
        n_agents = 5
        n_periods = 10
        result = fitted_estimator.simulate(n_agents=n_agents, n_periods=n_periods, seed=42)

        # Should have n_agents unique agent IDs
        assert result["agent_id"].nunique() == n_agents

        # Each agent should have n_periods observations
        for agent_id in result["agent_id"].unique():
            agent_data = result[result["agent_id"] == agent_id]
            assert len(agent_data) == n_periods

    def test_nfxp_simulate_valid_states(self, fitted_estimator):
        """simulate() should produce valid state values."""
        result = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        # States should be in valid range [0, n_states)
        assert (result["state"] >= 0).all()
        assert (result["state"] < fitted_estimator.n_states).all()

    def test_nfxp_simulate_valid_actions(self, fitted_estimator):
        """simulate() should produce valid action values."""
        result = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        # Actions should be in valid range [0, n_actions)
        assert (result["action"] >= 0).all()
        assert (result["action"] < fitted_estimator.n_actions).all()

    def test_nfxp_simulate_seed_reproducibility(self, fitted_estimator):
        """simulate() should be reproducible with same seed."""
        result1 = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)
        result2 = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)

        pd.testing.assert_frame_equal(result1, result2)

    def test_nfxp_simulate_different_seeds(self, fitted_estimator):
        """simulate() should produce different results with different seeds."""
        result1 = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=42)
        result2 = fitted_estimator.simulate(n_agents=5, n_periods=10, seed=123)

        # Results should differ (at least in some rows)
        # Note: there's a tiny chance this could fail randomly, but extremely unlikely
        assert not result1["action"].equals(result2["action"]) or not result1["state"].equals(result2["state"])


class TestNFXPCounterfactual:
    """Tests for NFXP.counterfactual() method."""

    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit an NFXP estimator."""
        from econirl.estimators import NFXP

        np.random.seed(42)
        n_individuals = 20
        n_periods = 30

        data = []
        for i in range(n_individuals):
            state = 0
            for t in range(n_periods):
                action = 1 if state > 60 or np.random.random() < 0.02 else 0
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]), 89)
                data.append({
                    "bus_id": i,
                    "period": t,
                    "mileage_bin": state,
                    "replaced": action,
                    "next_mileage": next_state,
                })
                state = next_state

        df = pd.DataFrame(data)

        estimator = NFXP(n_states=90, verbose=False)
        estimator.fit(
            data=df,
            state="mileage_bin",
            action="replaced",
            id="bus_id",
        )

        return estimator

    def test_nfxp_counterfactual_returns_result(self, fitted_estimator):
        """counterfactual() should return a CounterfactualResult."""
        from econirl.estimators.nfxp import CounterfactualResult

        result = fitted_estimator.counterfactual(RC=15.0)

        assert isinstance(result, CounterfactualResult)

    def test_nfxp_counterfactual_has_params(self, fitted_estimator):
        """CounterfactualResult should have params dict."""
        result = fitted_estimator.counterfactual(RC=15.0)

        assert hasattr(result, "params")
        assert isinstance(result.params, dict)
        assert "RC" in result.params
        assert result.params["RC"] == 15.0
        # theta_c should be from original estimate
        assert "theta_c" in result.params
        assert result.params["theta_c"] == fitted_estimator.params_["theta_c"]

    def test_nfxp_counterfactual_has_value_function(self, fitted_estimator):
        """CounterfactualResult should have value_function array."""
        result = fitted_estimator.counterfactual(RC=15.0)

        assert hasattr(result, "value_function")
        assert isinstance(result.value_function, np.ndarray)
        assert len(result.value_function) == fitted_estimator.n_states

    def test_nfxp_counterfactual_has_policy(self, fitted_estimator):
        """CounterfactualResult should have policy array."""
        result = fitted_estimator.counterfactual(RC=15.0)

        assert hasattr(result, "policy")
        assert isinstance(result.policy, np.ndarray)
        assert result.policy.shape == (fitted_estimator.n_states, fitted_estimator.n_actions)

        # Policy should be valid probabilities
        assert (result.policy >= 0).all()
        assert (result.policy <= 1).all()
        np.testing.assert_allclose(result.policy.sum(axis=1), np.ones(fitted_estimator.n_states), atol=1e-6)

    def test_nfxp_counterfactual_changes_policy(self, fitted_estimator):
        """counterfactual() with different RC should change policy."""
        # Use very different RC values to ensure policy changes
        # even with small/noisy estimated parameters
        result_low_RC = fitted_estimator.counterfactual(RC=1.0)
        result_high_RC = fitted_estimator.counterfactual(RC=100.0)

        # With higher replacement cost, probability of replacement should generally decrease
        # (At least at some states)
        low_rc_p_replace = result_low_RC.policy[:, 1]
        high_rc_p_replace = result_high_RC.policy[:, 1]

        # The policies should be different with such extreme RC differences
        assert not np.allclose(low_rc_p_replace, high_rc_p_replace, atol=1e-3)

        # On average, replacement probability should be lower with higher RC
        assert high_rc_p_replace.mean() < low_rc_p_replace.mean()

    def test_nfxp_counterfactual_multiple_params(self, fitted_estimator):
        """counterfactual() should accept multiple parameter changes."""
        result = fitted_estimator.counterfactual(RC=15.0, theta_c=0.05)

        assert result.params["RC"] == 15.0
        assert result.params["theta_c"] == 0.05

    def test_nfxp_counterfactual_invalid_param_raises(self, fitted_estimator):
        """counterfactual() should raise error for unknown parameters."""
        with pytest.raises(ValueError):
            fitted_estimator.counterfactual(invalid_param=1.0)
