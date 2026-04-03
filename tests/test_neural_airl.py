"""Tests for NeuralAIRL estimator.

Tests cover:
- Basic fit without features
- Fit with feature projection
- Context conditioning
- Policy and value shapes
- EstimatorProtocol conformance
- predict_proba and predict_reward
- Confidence intervals
- Summary output
- Custom encoders
- Projection R-squared range
"""

import numpy as np
import pandas as pd
import pytest
import torch

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import TrajectoryPanel
from econirl.estimators.neural_airl import NeuralAIRL
from econirl.estimators.protocol import EstimatorProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_data():
    """Small 10-state, 3-action dataset with 50 agents x 20 periods."""
    np.random.seed(42)
    records = []
    for uid in range(50):
        state = 0
        dest = np.random.randint(5, 10)  # destination context
        for t in range(20):
            action = np.random.randint(3)
            next_state = min(state + action, 9)
            records.append(
                {
                    "id": uid,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "dest": dest,
                }
            )
            state = next_state
    return pd.DataFrame(records)


@pytest.fixture
def small_features():
    """Feature matrix (10 states, 3 actions, 2 features) for projection."""
    n_states, n_actions, n_features = 10, 3, 2
    features = torch.zeros((n_states, n_actions, n_features))
    # Feature 0: state index (normalized)
    for s in range(n_states):
        features[s, :, 0] = -s / 9.0
    # Feature 1: action cost (action 0 = 0, action 1 = -0.5, action 2 = -1)
    for a in range(n_actions):
        features[:, a, 1] = -a / 2.0
    return RewardSpec(features, names=["state_cost", "action_cost"])


@pytest.fixture
def fitted_model(small_data):
    """A fitted NeuralAIRL model without features."""
    model = NeuralAIRL(
        n_actions=3,
        discount=0.95,
        max_epochs=30,
        patience=10,
        reward_hidden_dim=32,
        reward_num_layers=2,
        shaping_hidden_dim=32,
        shaping_num_layers=2,
        policy_hidden_dim=32,
        policy_num_layers=2,
        batch_size=256,
        disc_steps=2,
    )
    model.fit(
        data=small_data,
        state="state",
        action="action",
        id="id",
    )
    return model


@pytest.fixture
def fitted_model_with_features(small_data, small_features):
    """A fitted NeuralAIRL model with feature projection."""
    model = NeuralAIRL(
        n_actions=3,
        discount=0.95,
        max_epochs=30,
        patience=10,
        reward_hidden_dim=32,
        reward_num_layers=2,
        shaping_hidden_dim=32,
        shaping_num_layers=2,
        policy_hidden_dim=32,
        policy_num_layers=2,
        batch_size=256,
        disc_steps=2,
    )
    model.fit(
        data=small_data,
        state="state",
        action="action",
        id="id",
        features=small_features,
    )
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFitBasic:
    """Test basic fit without features."""

    def test_fit_accepts_trajectory_panel(self, small_data):
        """fit() should work when input data is a TrajectoryPanel."""
        panel = TrajectoryPanel.from_dataframe(
            small_data, state="state", action="action", id="id"
        )
        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            batch_size=256,
            disc_steps=1,
        )
        returned = model.fit(
            data=panel,
        )
        assert returned is model
        assert model.policy_ is not None

    def test_fit_runs_without_error(self, fitted_model):
        """Fit should complete without raising."""
        assert fitted_model.policy_ is not None

    def test_params_none_without_features(self, fitted_model):
        """Without features, params_ should be None."""
        assert fitted_model.params_ is None
        assert fitted_model.se_ is None
        assert fitted_model.pvalues_ is None
        assert fitted_model.projection_r2_ is None
        assert fitted_model.coef_ is None

    def test_converged_is_set(self, fitted_model):
        """converged_ should be a boolean after fitting."""
        assert isinstance(fitted_model.converged_, bool)

    def test_n_epochs_is_set(self, fitted_model):
        """n_epochs_ should be a positive integer after fitting."""
        assert isinstance(fitted_model.n_epochs_, int)
        assert fitted_model.n_epochs_ > 0


class TestFitWithFeatures:
    """Test fit with feature projection."""

    def test_params_has_correct_keys(self, fitted_model_with_features):
        """params_ should have the correct feature names."""
        assert fitted_model_with_features.params_ is not None
        assert set(fitted_model_with_features.params_.keys()) == {
            "state_cost",
            "action_cost",
        }

    def test_se_has_correct_keys(self, fitted_model_with_features):
        """se_ should have the same keys as params_."""
        assert fitted_model_with_features.se_ is not None
        assert set(fitted_model_with_features.se_.keys()) == {
            "state_cost",
            "action_cost",
        }

    def test_pvalues_has_correct_keys(self, fitted_model_with_features):
        """pvalues_ should have the same keys as params_."""
        assert fitted_model_with_features.pvalues_ is not None
        assert set(fitted_model_with_features.pvalues_.keys()) == {
            "state_cost",
            "action_cost",
        }

    def test_projection_r2_is_float(self, fitted_model_with_features):
        """projection_r2_ should be a float."""
        assert isinstance(fitted_model_with_features.projection_r2_, float)

    def test_coef_array(self, fitted_model_with_features):
        """coef_ should be a numpy array with correct length."""
        assert fitted_model_with_features.coef_ is not None
        assert isinstance(fitted_model_with_features.coef_, np.ndarray)
        assert len(fitted_model_with_features.coef_) == 2


class TestFitWithContext:
    """Test fit with context conditioning."""

    def test_context_column(self, small_data):
        """Passing a context column name should work."""
        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=20,
            patience=10,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            batch_size=256,
            disc_steps=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            context="dest",
        )
        assert model.policy_ is not None

    def test_context_tensor(self, small_data):
        """Passing a context tensor should work."""
        N = len(small_data)
        ctx = torch.randint(0, 5, (N,))
        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=20,
            patience=10,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            batch_size=256,
            disc_steps=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            context=ctx,
        )
        assert model.policy_ is not None


class TestPolicyAndValue:
    """Test policy and value function shapes."""

    def test_policy_shape(self, fitted_model):
        """Policy should be (n_states, n_actions)."""
        assert fitted_model.policy_.shape == (10, 3)

    def test_policy_sums_to_one(self, fitted_model):
        """Each row of policy should sum to 1."""
        row_sums = fitted_model.policy_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_policy_nonnegative(self, fitted_model):
        """Policy probabilities should be non-negative."""
        assert (fitted_model.policy_ >= 0).all()

    def test_value_shape(self, fitted_model):
        """Value function should be (n_states,)."""
        assert fitted_model.value_.shape == (10,)


class TestProtocol:
    """Test EstimatorProtocol conformance."""

    def test_protocol_conformance(self, fitted_model_with_features):
        """Model should satisfy EstimatorProtocol."""
        assert isinstance(fitted_model_with_features, EstimatorProtocol)


class TestPredictProba:
    """Test predict_proba method."""

    def test_correct_shape(self, fitted_model):
        """predict_proba should return (len(states), n_actions)."""
        states = np.array([0, 3, 7])
        proba = fitted_model.predict_proba(states)
        assert proba.shape == (3, 3)

    def test_sums_to_one(self, fitted_model):
        """Probabilities should sum to 1."""
        states = np.array([0, 5, 9])
        proba = fitted_model.predict_proba(states)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_not_fitted_raises(self):
        """predict_proba should raise if model is not fitted."""
        model = NeuralAIRL()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.array([0]))


class TestPredictReward:
    """Test predict_reward method."""

    def test_returns_correct_shape(self, fitted_model):
        """predict_reward should return (N,) tensor."""
        states = torch.tensor([0, 3, 7])
        actions = torch.tensor([0, 1, 2])
        rewards = fitted_model.predict_reward(states, actions)
        assert rewards.shape == (3,)

    def test_returns_tensor(self, fitted_model):
        """predict_reward should return a tensor."""
        states = torch.tensor([0, 5])
        actions = torch.tensor([1, 0])
        rewards = fitted_model.predict_reward(states, actions)
        assert isinstance(rewards, torch.Tensor)

    def test_with_context(self, fitted_model):
        """predict_reward should accept context argument."""
        states = torch.tensor([0, 5])
        actions = torch.tensor([1, 0])
        contexts = torch.tensor([0, 3])
        rewards = fitted_model.predict_reward(states, actions, contexts)
        assert rewards.shape == (2,)

    def test_not_fitted_raises(self):
        """predict_reward should raise if not fitted."""
        model = NeuralAIRL()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_reward(torch.tensor([0]), torch.tensor([0]))


class TestConfInt:
    """Test confidence intervals."""

    def test_returns_dict_with_intervals(self, fitted_model_with_features):
        """conf_int should return dict of (lower, upper) tuples."""
        ci = fitted_model_with_features.conf_int(alpha=0.05)
        assert isinstance(ci, dict)
        assert set(ci.keys()) == {"state_cost", "action_cost"}
        for name, (lo, hi) in ci.items():
            assert isinstance(lo, float)
            assert isinstance(hi, float)
            # Lower should be less than upper (for finite SEs)
            if np.isfinite(lo) and np.isfinite(hi):
                assert lo <= hi

    def test_raises_without_features(self, fitted_model):
        """conf_int should raise if no features were provided."""
        with pytest.raises(RuntimeError, match="No projected parameters"):
            fitted_model.conf_int()


class TestSummary:
    """Test summary output."""

    def test_returns_nonempty_string(self, fitted_model_with_features):
        """summary() should return a non-empty string."""
        s = fitted_model_with_features.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_method_name(self, fitted_model_with_features):
        """Summary should mention the method name."""
        s = fitted_model_with_features.summary()
        assert "NeuralAIRL" in s

    def test_contains_r2(self, fitted_model_with_features):
        """Summary should include R2 info."""
        s = fitted_model_with_features.summary()
        assert "R2" in s

    def test_contains_parameter_names(self, fitted_model_with_features):
        """Summary should include the parameter names."""
        s = fitted_model_with_features.summary()
        assert "state_cost" in s
        assert "action_cost" in s

    def test_not_fitted_message(self):
        """Summary should indicate not fitted for unfitted model."""
        model = NeuralAIRL()
        s = model.summary()
        assert "Not fitted" in s

    def test_summary_without_features(self, fitted_model):
        """Summary without features should mention no projection."""
        s = fitted_model.summary()
        assert "No feature projection" in s or "None" in s


class TestNoTransitionMatrix:
    """Test that transitions= is accepted but ignored."""

    def test_transitions_ignored(self, small_data):
        """Passing transitions= should not raise."""
        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=10,
            patience=5,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            disc_steps=1,
        )
        # Pass a dummy transitions argument
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            transitions=np.eye(10),
        )
        assert model.policy_ is not None


class TestCustomEncoders:
    """Test custom state and context encoders."""

    def test_custom_context_encoder(self, small_data):
        """Custom context encoder should be used."""
        # One-hot context encoder for 10 possible contexts
        def ctx_encoder(c):
            return torch.nn.functional.one_hot(c, 10).float()

        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            context_encoder=ctx_encoder,
            context_dim=10,
            disc_steps=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            context="dest",
        )
        assert model.policy_ is not None

    def test_custom_state_encoder(self, small_data):
        """Custom state encoder should be used."""
        # One-hot state encoder
        def state_encoder(s):
            return torch.nn.functional.one_hot(s, 10).float()

        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            state_encoder=state_encoder,
            state_dim=10,
            disc_steps=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
        )
        assert model.policy_ is not None


class TestProjectionR2:
    """Test projection R-squared range."""

    def test_r2_in_valid_range(self, fitted_model_with_features):
        """Projection R2 should be at most 1."""
        r2 = fitted_model_with_features.projection_r2_
        assert r2 <= 1.0 + 1e-6

    def test_r2_is_finite(self, fitted_model_with_features):
        """Projection R2 should be finite."""
        assert np.isfinite(fitted_model_with_features.projection_r2_)


class TestRepr:
    """Test __repr__ output."""

    def test_unfitted_repr(self):
        """Unfitted model repr should say fitted=False."""
        model = NeuralAIRL(n_actions=5)
        r = repr(model)
        assert "fitted=False" in r
        assert "n_actions=5" in r

    def test_fitted_repr(self, fitted_model):
        """Fitted model repr should say fitted=True."""
        r = repr(fitted_model)
        assert "fitted=True" in r


class TestRawTensorFeatures:
    """Test using raw tensor features (not RewardSpec)."""

    def test_raw_tensor_features(self, small_data):
        """Should accept raw (S, A, K) tensor as features."""
        features = torch.randn(10, 3, 2)
        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            feature_names=["f0", "f1"],
            disc_steps=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            features=features,
        )
        assert model.params_ is not None
        assert set(model.params_.keys()) == {"f0", "f1"}

    def test_auto_feature_names(self, small_data):
        """Without feature_names, should auto-generate f0, f1, ..."""
        features = torch.randn(10, 3, 3)
        model = NeuralAIRL(
            n_actions=3,
            discount=0.95,
            max_epochs=15,
            patience=5,
            reward_hidden_dim=16,
            reward_num_layers=1,
            shaping_hidden_dim=16,
            shaping_num_layers=1,
            policy_hidden_dim=16,
            policy_num_layers=1,
            disc_steps=1,
        )
        model.fit(
            data=small_data,
            state="state",
            action="action",
            id="id",
            features=features,
        )
        assert set(model.params_.keys()) == {"f0", "f1", "f2"}
