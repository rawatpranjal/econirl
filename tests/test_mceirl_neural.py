"""Tests for MCEIRLNeural estimator.

Tests that MCEIRLNeural:
1. Basic fit with transitions
2. params_ populated when features provided
3. projection_r2_ is float
4. policy_ shape correct
5. EstimatorProtocol conformance
6. predict_proba() works
7. summary() returns string
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from econirl.core.reward_spec import RewardSpec
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.estimators.protocol import EstimatorProtocol


# ---------------------------------------------------------------------------
# Fixtures: small 5-state gridworld
# ---------------------------------------------------------------------------

_N_STATES = 5
_N_ACTIONS = 2
_DISCOUNT = 0.9


def _make_gridworld_transitions(
    n_states: int = _N_STATES,
    n_actions: int = _N_ACTIONS,
) -> torch.Tensor:
    """Create simple deterministic-ish transitions.

    Action 0: move right (state+1, wrapping at end)
    Action 1: stay in place
    """
    T = torch.zeros(n_actions, n_states, n_states, dtype=torch.float32)

    for s in range(n_states):
        # Action 0: move right with some noise
        next_s = (s + 1) % n_states
        T[0, s, next_s] = 0.9
        T[0, s, s] = 0.1

        # Action 1: stay
        T[1, s, s] = 0.9
        T[1, s, (s + 1) % n_states] = 0.1

    return T


def _make_gridworld_data(
    n_states: int = _N_STATES,
    n_actions: int = _N_ACTIONS,
    n_individuals: int = 20,
    n_periods: int = 30,
    seed: int = 42,
):
    """Generate synthetic gridworld data as a DataFrame."""
    import pandas as pd

    np.random.seed(seed)
    T = _make_gridworld_transitions(n_states, n_actions).numpy()
    data = []

    for i in range(n_individuals):
        state = np.random.randint(n_states)
        for t in range(n_periods):
            # Simple policy: prefer action 0 (move) at low states,
            # prefer action 1 (stay) at high states
            p_action0 = 0.8 if state < n_states // 2 else 0.2
            action = 0 if np.random.random() < p_action0 else 1
            next_state = np.random.choice(
                n_states, p=T[action, state, :]
            )
            data.append(
                {
                    "agent_id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                }
            )
            state = next_state

    return pd.DataFrame(data)


def _make_features(n_states: int = _N_STATES) -> RewardSpec:
    """Create state features for projection."""
    s = torch.arange(n_states, dtype=torch.float32)
    state_features = torch.stack([s / n_states, (s / n_states) ** 2], dim=1)
    return RewardSpec(
        state_features, names=["linear", "quadratic"], n_actions=_N_ACTIONS
    )


@pytest.fixture(scope="module")
def transitions():
    return _make_gridworld_transitions()


@pytest.fixture(scope="module")
def gridworld_df():
    return _make_gridworld_data()


@pytest.fixture(scope="module")
def fitted_model(gridworld_df, transitions):
    """Fitted MCEIRLNeural model shared across test classes (state reward)."""
    features = _make_features()
    model = MCEIRLNeural(
        n_states=_N_STATES,
        n_actions=_N_ACTIONS,
        discount=_DISCOUNT,
        reward_type="state",
        max_epochs=100,
        lr=1e-2,
        reward_hidden_dim=32,
        reward_num_layers=1,
        verbose=False,
    )
    model.fit(
        gridworld_df,
        state="state",
        action="action",
        id="agent_id",
        transitions=transitions,
        features=features,
    )
    return model


@pytest.fixture(scope="module")
def fitted_model_no_features(gridworld_df, transitions):
    """Fitted MCEIRLNeural without feature projection (state reward)."""
    model = MCEIRLNeural(
        n_states=_N_STATES,
        n_actions=_N_ACTIONS,
        discount=_DISCOUNT,
        reward_type="state",
        max_epochs=50,
        lr=1e-2,
        reward_hidden_dim=32,
        reward_num_layers=1,
        verbose=False,
    )
    model.fit(
        gridworld_df,
        state="state",
        action="action",
        id="agent_id",
        transitions=transitions,
    )
    return model


@pytest.fixture(scope="module")
def fitted_model_state_action(gridworld_df, transitions):
    """Fitted MCEIRLNeural with reward_type='state_action'."""
    features = _make_features()
    model = MCEIRLNeural(
        n_states=_N_STATES,
        n_actions=_N_ACTIONS,
        discount=_DISCOUNT,
        reward_type="state_action",
        max_epochs=100,
        lr=1e-2,
        reward_hidden_dim=32,
        reward_num_layers=1,
        verbose=False,
    )
    model.fit(
        gridworld_df,
        state="state",
        action="action",
        id="agent_id",
        transitions=transitions,
        features=features,
    )
    return model


# ---------------------------------------------------------------------------
# 1. Basic fit with transitions
# ---------------------------------------------------------------------------


class TestBasicFit:
    """MCEIRLNeural fits with transitions provided."""

    def test_fit_returns_self(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state",
            max_epochs=10,
            verbose=False,
        )
        result = model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )
        assert result is model

    def test_requires_transitions(self, gridworld_df):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        with pytest.raises(ValueError, match="requires transitions"):
            model.fit(
                gridworld_df,
                state="state",
                action="action",
                id="agent_id",
            )

    def test_reward_populated(self, fitted_model_no_features):
        assert fitted_model_no_features.reward_ is not None
        assert isinstance(fitted_model_no_features.reward_, np.ndarray)
        assert fitted_model_no_features.reward_.shape == (_N_STATES,)

    def test_converged_is_bool(self, fitted_model_no_features):
        assert fitted_model_no_features.converged_ is not None
        assert isinstance(fitted_model_no_features.converged_, bool)

    def test_n_epochs_positive(self, fitted_model_no_features):
        assert fitted_model_no_features.n_epochs_ is not None
        assert fitted_model_no_features.n_epochs_ > 0


# ---------------------------------------------------------------------------
# 2. params_ populated when features provided
# ---------------------------------------------------------------------------


class TestParamsWithFeatures:
    """params_ populated when features are provided."""

    def test_params_present(self, fitted_model):
        assert fitted_model.params_ is not None
        assert "linear" in fitted_model.params_
        assert "quadratic" in fitted_model.params_

    def test_se_present(self, fitted_model):
        assert fitted_model.se_ is not None
        assert "linear" in fitted_model.se_

    def test_pvalues_present(self, fitted_model):
        assert fitted_model.pvalues_ is not None
        assert "linear" in fitted_model.pvalues_

    def test_coef_present(self, fitted_model):
        assert fitted_model.coef_ is not None
        assert isinstance(fitted_model.coef_, np.ndarray)
        assert len(fitted_model.coef_) == 2

    def test_params_none_without_features(self, fitted_model_no_features):
        assert fitted_model_no_features.params_ is None
        assert fitted_model_no_features.se_ is None
        assert fitted_model_no_features.pvalues_ is None


# ---------------------------------------------------------------------------
# 3. projection_r2_ is float
# ---------------------------------------------------------------------------


class TestProjectionR2:
    """projection_r2_ is a float when features are provided."""

    def test_r2_is_float(self, fitted_model):
        assert fitted_model.projection_r2_ is not None
        assert isinstance(fitted_model.projection_r2_, float)

    def test_r2_none_without_features(self, fitted_model_no_features):
        assert fitted_model_no_features.projection_r2_ is None


# ---------------------------------------------------------------------------
# 4. policy_ shape correct
# ---------------------------------------------------------------------------


class TestPolicyShape:
    """policy_ has correct shape."""

    def test_policy_shape(self, fitted_model):
        assert fitted_model.policy_ is not None
        assert fitted_model.policy_.shape == (_N_STATES, _N_ACTIONS)

    def test_policy_valid_probabilities(self, fitted_model):
        assert (fitted_model.policy_ >= 0).all()
        assert (fitted_model.policy_ <= 1).all()
        row_sums = fitted_model.policy_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(_N_STATES), atol=1e-6)

    def test_value_shape(self, fitted_model):
        assert fitted_model.value_ is not None
        assert fitted_model.value_.shape == (_N_STATES,)


# ---------------------------------------------------------------------------
# 5. EstimatorProtocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """MCEIRLNeural satisfies the EstimatorProtocol."""

    def test_satisfies_protocol(self, fitted_model):
        assert isinstance(fitted_model, EstimatorProtocol)

    def test_unfitted_has_protocol_attributes(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
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
# 6. predict_proba() works
# ---------------------------------------------------------------------------


class TestPredictProba:
    """predict_proba() returns valid probabilities."""

    def test_predict_proba_shape(self, fitted_model):
        proba = fitted_model.predict_proba(np.array([0, 2, 4]))
        assert proba.shape == (3, _N_ACTIONS)

    def test_predict_proba_valid(self, fitted_model):
        proba = fitted_model.predict_proba(np.array([0, 1, 2]))
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)

    def test_predict_proba_unfitted_raises(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0]))


# ---------------------------------------------------------------------------
# 7. summary() returns string
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns a non-empty string."""

    def test_summary_returns_string(self, fitted_model):
        summary = fitted_model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "MCEIRLNeural" in summary

    def test_summary_unfitted(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        summary = model.summary()
        assert isinstance(summary, str)
        assert "Not fitted" in summary

    def test_repr(self, fitted_model):
        r = repr(fitted_model)
        assert "MCEIRLNeural" in r
        assert "fitted=True" in r

    def test_repr_unfitted(self):
        r = repr(MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS))
        assert "fitted=False" in r


# ---------------------------------------------------------------------------
# 8. conf_int() works with features
# ---------------------------------------------------------------------------


class TestConfInt:
    """conf_int() returns valid intervals when features are provided."""

    def test_conf_int_keys(self, fitted_model):
        ci = fitted_model.conf_int()
        assert "linear" in ci
        assert "quadratic" in ci

    def test_conf_int_brackets_estimate(self, fitted_model):
        ci = fitted_model.conf_int()
        for name in fitted_model.params_:
            lower, upper = ci[name]
            est = fitted_model.params_[name]
            if np.isfinite(lower) and np.isfinite(upper):
                assert lower <= est <= upper, (
                    f"CI for {name}: ({lower}, {upper}) does not contain "
                    f"estimate {est}"
                )

    def test_conf_int_no_features_raises(self, fitted_model_no_features):
        with pytest.raises(RuntimeError, match="No projected parameters"):
            fitted_model_no_features.conf_int()


# ---------------------------------------------------------------------------
# 9. reward_type="state_action" tests
# ---------------------------------------------------------------------------


class TestStateActionRewardType:
    """Tests for reward_type='state_action' (R(s,a) network)."""

    def test_reward_shape_state_action(self, fitted_model_state_action):
        """R(s,a) reward should be (n_states, n_actions)."""
        assert fitted_model_state_action.reward_ is not None
        assert fitted_model_state_action.reward_.shape == (
            _N_STATES,
            _N_ACTIONS,
        )

    def test_policy_shape_state_action(self, fitted_model_state_action):
        """policy_ shape unchanged regardless of reward_type."""
        assert fitted_model_state_action.policy_ is not None
        assert fitted_model_state_action.policy_.shape == (
            _N_STATES,
            _N_ACTIONS,
        )

    def test_policy_valid_probabilities_state_action(
        self, fitted_model_state_action
    ):
        policy = fitted_model_state_action.policy_
        assert (policy >= 0).all()
        assert (policy <= 1).all()
        row_sums = policy.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(_N_STATES), atol=1e-6)

    def test_params_populated_state_action(self, fitted_model_state_action):
        """params_ populated when features provided with state_action."""
        assert fitted_model_state_action.params_ is not None
        assert "linear" in fitted_model_state_action.params_
        assert "quadratic" in fitted_model_state_action.params_

    def test_projection_r2_state_action(self, fitted_model_state_action):
        assert fitted_model_state_action.projection_r2_ is not None
        assert isinstance(fitted_model_state_action.projection_r2_, float)

    def test_value_shape_state_action(self, fitted_model_state_action):
        assert fitted_model_state_action.value_ is not None
        assert fitted_model_state_action.value_.shape == (_N_STATES,)

    def test_predict_proba_state_action(self, fitted_model_state_action):
        proba = fitted_model_state_action.predict_proba(np.array([0, 2, 4]))
        assert proba.shape == (3, _N_ACTIONS)
        assert (proba >= 0).all()
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-6)

    def test_summary_state_action(self, fitted_model_state_action):
        summary = fitted_model_state_action.summary()
        assert isinstance(summary, str)
        assert "MCEIRLNeural" in summary
        assert "state_action" in summary

    def test_conf_int_state_action(self, fitted_model_state_action):
        ci = fitted_model_state_action.conf_int()
        assert "linear" in ci
        for name in fitted_model_state_action.params_:
            lower, upper = ci[name]
            est = fitted_model_state_action.params_[name]
            if np.isfinite(lower) and np.isfinite(upper):
                assert lower <= est <= upper

    def test_invalid_reward_type_raises(self):
        with pytest.raises(ValueError, match="reward_type"):
            MCEIRLNeural(
                n_states=_N_STATES,
                n_actions=_N_ACTIONS,
                reward_type="invalid",
            )

    def test_default_reward_type_is_state_action(self):
        model = MCEIRLNeural(n_states=_N_STATES, n_actions=_N_ACTIONS)
        assert model.reward_type == "state_action"

    def test_state_action_fit_returns_self(self, gridworld_df, transitions):
        model = MCEIRLNeural(
            n_states=_N_STATES,
            n_actions=_N_ACTIONS,
            discount=_DISCOUNT,
            reward_type="state_action",
            max_epochs=10,
            verbose=False,
        )
        result = model.fit(
            gridworld_df,
            state="state",
            action="action",
            id="agent_id",
            transitions=transitions,
        )
        assert result is model
        # Without features, reward_ should still be (S, A)
        assert result.reward_.shape == (_N_STATES, _N_ACTIONS)
