"""Tests for RewardSpec — unified feature specification."""

from __future__ import annotations

import pytest
import torch

from econirl.core.reward_spec import RewardSpec
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.preferences.reward import LinearReward


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sak_features():
    """(S=4, A=2, K=3) feature tensor with known values."""
    torch.manual_seed(42)
    return torch.randn(4, 2, 3)


@pytest.fixture
def sk_features():
    """(S=4, K=3) state-only feature tensor."""
    torch.manual_seed(99)
    return torch.randn(4, 3)


@pytest.fixture
def names():
    return ["cost", "benefit", "distance"]


@pytest.fixture
def params():
    return torch.tensor([1.0, -0.5, 0.3])


# ------------------------------------------------------------------ #
# 1. Construct from (S, A, K) tensor
# ------------------------------------------------------------------ #

class TestConstructFromSAK:
    def test_shape(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        assert spec.feature_matrix.shape == (4, 2, 3)
        assert spec.num_states == 4
        assert spec.num_actions == 2
        assert spec.num_parameters == 3

    def test_compute_output_shape(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        R = spec.compute(params)
        assert R.shape == (4, 2)

    def test_is_not_state_only(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        assert spec.is_state_only is False


# ------------------------------------------------------------------ #
# 2. Construct from (S, K) tensor — auto-broadcast
# ------------------------------------------------------------------ #

class TestConstructFromSK:
    def test_broadcast_shape(self, sk_features, names):
        spec = RewardSpec(sk_features, names=names, n_actions=2)
        assert spec.feature_matrix.shape == (4, 2, 3)

    def test_same_features_all_actions(self, sk_features, names):
        spec = RewardSpec(sk_features, names=names, n_actions=3)
        fm = spec.feature_matrix
        # All actions should have the same features
        for a in range(3):
            assert torch.allclose(fm[:, a, :], fm[:, 0, :])

    def test_is_state_only(self, sk_features, names):
        spec = RewardSpec(sk_features, names=names, n_actions=2)
        assert spec.is_state_only is True

    def test_missing_n_actions_raises(self, sk_features, names):
        with pytest.raises(ValueError, match="n_actions is required"):
            RewardSpec(sk_features, names=names)

    def test_wrong_ndim_raises(self, names):
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            RewardSpec(torch.randn(5), names=names)


# ------------------------------------------------------------------ #
# 3. state_dependent() classmethod
# ------------------------------------------------------------------ #

class TestStateDependentClassmethod:
    def test_same_as_sk_constructor(self, sk_features, names):
        spec_a = RewardSpec(sk_features, names=names, n_actions=2)
        spec_b = RewardSpec.state_dependent(sk_features, names=names, n_actions=2)
        assert torch.allclose(spec_a.feature_matrix, spec_b.feature_matrix)
        assert spec_b.is_state_only is True

    def test_rejects_3d(self, sak_features, names):
        with pytest.raises(ValueError, match="must be 2D"):
            RewardSpec.state_dependent(sak_features, names=names, n_actions=2)


# ------------------------------------------------------------------ #
# 4. state_action_dependent() classmethod
# ------------------------------------------------------------------ #

class TestStateActionDependentClassmethod:
    def test_same_as_sak_constructor(self, sak_features, names):
        spec_a = RewardSpec(sak_features, names=names)
        spec_b = RewardSpec.state_action_dependent(sak_features, names=names)
        assert torch.allclose(spec_a.feature_matrix, spec_b.feature_matrix)
        assert spec_b.is_state_only is False

    def test_rejects_2d(self, sk_features, names):
        with pytest.raises(ValueError, match="must be 3D"):
            RewardSpec.state_action_dependent(sk_features, names=names)


# ------------------------------------------------------------------ #
# 5. compute(params) — numerical correctness
# ------------------------------------------------------------------ #

class TestCompute:
    def test_known_values(self):
        """R[s,a] = sum_k params[k] * features[s,a,k]."""
        features = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],  # s=0
            [[2.0, 3.0], [4.0, 5.0]],  # s=1
        ])  # (S=2, A=2, K=2)
        params = torch.tensor([1.0, 2.0])
        spec = RewardSpec(features, names=["a", "b"])
        R = spec.compute(params)

        # s=0, a=0: 1*1 + 0*2 = 1
        # s=0, a=1: 0*1 + 1*2 = 2
        # s=1, a=0: 2*1 + 3*2 = 8
        # s=1, a=1: 4*1 + 5*2 = 14
        expected = torch.tensor([[1.0, 2.0], [8.0, 14.0]])
        assert torch.allclose(R, expected)

    def test_matches_einsum(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        R = spec.compute(params)
        expected = torch.einsum("sak,k->sa", sak_features, params)
        assert torch.allclose(R, expected)


# ------------------------------------------------------------------ #
# 6. compute_gradient()
# ------------------------------------------------------------------ #

class TestComputeGradient:
    def test_returns_feature_matrix(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        grad = spec.compute_gradient(params)
        assert grad.shape == (4, 2, 3)
        assert torch.allclose(grad, spec.feature_matrix)

    def test_is_independent_of_params(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        g1 = spec.compute_gradient(torch.ones(3))
        g2 = spec.compute_gradient(torch.zeros(3))
        assert torch.allclose(g1, g2)


# ------------------------------------------------------------------ #
# 7. compute_hessian()
# ------------------------------------------------------------------ #

class TestComputeHessian:
    def test_all_zeros(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        H = spec.compute_hessian(params)
        assert H.shape == (4, 2, 3, 3)
        assert (H == 0).all()


# ------------------------------------------------------------------ #
# 8. to_linear_utility()
# ------------------------------------------------------------------ #

class TestToLinearUtility:
    def test_same_compute_output(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        lu = spec.to_linear_utility()

        assert isinstance(lu, LinearUtility)
        assert torch.allclose(spec.compute(params), lu.compute(params))
        assert lu.parameter_names == spec.parameter_names

    def test_feature_matrix_matches(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        lu = spec.to_linear_utility()
        assert torch.allclose(spec.feature_matrix, lu.feature_matrix)


# ------------------------------------------------------------------ #
# 9. to_action_dependent_reward()
# ------------------------------------------------------------------ #

class TestToActionDependentReward:
    def test_same_compute_output(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        adr = spec.to_action_dependent_reward()

        assert isinstance(adr, ActionDependentReward)
        assert torch.allclose(spec.compute(params), adr.compute(params))
        assert adr.parameter_names == spec.parameter_names


# ------------------------------------------------------------------ #
# 10. to_linear_reward()
# ------------------------------------------------------------------ #

class TestToLinearReward:
    def test_works_for_state_only(self, sk_features, names, params):
        spec = RewardSpec(sk_features, names=names, n_actions=2)
        lr = spec.to_linear_reward()

        assert isinstance(lr, LinearReward)
        assert torch.allclose(spec.compute(params), lr.compute(params))
        assert lr.parameter_names == spec.parameter_names

    def test_raises_for_action_dependent(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        with pytest.raises(ValueError, match="features differ across actions"):
            spec.to_linear_reward()

    def test_works_for_identical_action_features(self, names, params):
        """Even a 3D tensor should convert if all actions are identical."""
        sk = torch.randn(5, 3)
        sak = sk.unsqueeze(1).expand(-1, 2, -1).clone()
        spec = RewardSpec(sak, names=names)
        lr = spec.to_linear_reward()
        assert isinstance(lr, LinearReward)
        assert torch.allclose(spec.compute(params), lr.compute(params))


# ------------------------------------------------------------------ #
# 11. validate_parameters()
# ------------------------------------------------------------------ #

class TestValidateParameters:
    def test_correct_shape_passes(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        spec.validate_parameters(params)  # should not raise

    def test_wrong_shape_raises(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        with pytest.raises(ValueError, match="Expected parameters"):
            spec.validate_parameters(torch.zeros(5))

    def test_2d_raises(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        with pytest.raises(ValueError, match="Expected parameters"):
            spec.validate_parameters(torch.zeros(3, 1))


# ------------------------------------------------------------------ #
# 12. subset_states()
# ------------------------------------------------------------------ #

class TestSubsetStates:
    def test_correct_subset(self, sak_features, names, params):
        spec = RewardSpec(sak_features, names=names)
        idx = torch.tensor([0, 2])
        sub = spec.subset_states(idx)

        assert sub.num_states == 2
        assert sub.num_actions == 2
        assert sub.num_parameters == 3

        # Check feature values match
        assert torch.allclose(sub.feature_matrix, sak_features[idx, :, :])

        # Check compute works on subset
        R_full = spec.compute(params)
        R_sub = sub.compute(params)
        assert torch.allclose(R_sub, R_full[idx, :])


# ------------------------------------------------------------------ #
# 13. to(device)
# ------------------------------------------------------------------ #

class TestToDevice:
    def test_cpu_to_cpu(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        spec_cpu = spec.to("cpu")

        assert spec_cpu.feature_matrix.device == torch.device("cpu")
        assert torch.allclose(spec.feature_matrix, spec_cpu.feature_matrix)
        assert spec_cpu.parameter_names == spec.parameter_names
        assert spec_cpu.is_state_only == spec.is_state_only

    def test_returns_new_instance(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        spec2 = spec.to("cpu")
        assert spec2 is not spec


# ------------------------------------------------------------------ #
# Additional edge-case tests
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_names_mismatch_raises(self):
        features = torch.randn(3, 2, 4)
        with pytest.raises(ValueError, match="names must have 4 elements"):
            RewardSpec(features, names=["a", "b"])

    def test_n_actions_conflict_raises(self):
        features = torch.randn(3, 2, 4)
        with pytest.raises(ValueError, match="n_actions=5 was also provided"):
            RewardSpec(features, names=["a", "b", "c", "d"], n_actions=5)

    def test_get_initial_parameters(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        init = spec.get_initial_parameters()
        assert init.shape == (3,)
        assert (init == 0).all()

    def test_get_parameter_bounds(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        lower, upper = spec.get_parameter_bounds()
        assert lower is None
        assert upper is None

    def test_repr(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        r = repr(spec)
        assert "RewardSpec" in r
        assert "num_states=4" in r
        assert "state_action" in r

    def test_parameter_names_is_copy(self, sak_features, names):
        spec = RewardSpec(sak_features, names=names)
        returned = spec.parameter_names
        returned.append("extra")
        assert len(spec.parameter_names) == 3  # original unchanged
