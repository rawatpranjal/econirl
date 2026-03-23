# Fix Parameter Recovery for NFXP and MCE IRL

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix NFXP and MCE IRL to correctly recover ground truth parameters on Rust bus engine data.

**Architecture:** Two root causes identified:
1. **NFXP inner loop never converges** - With β=0.9999, value iteration needs ~276,000 iterations but defaults to 1,000. Parameters are wrong because policy is wrong.
2. **MCE IRL uses state-only features** - Rust model has action-dependent utilities (operating cost vs replacement cost), but LinearReward broadcasts same reward to all actions.

**Tech Stack:** PyTorch, econirl existing infrastructure

---

### Task 1: Fix NFXP Inner Loop Convergence

**Files:**
- Test: `tests/test_nfxp_convergence.py`
- Modify: `src/econirl/estimation/nfxp.py`

**Step 1: Write test demonstrating the problem**

```python
# tests/test_nfxp_convergence.py
"""Test NFXP inner loop convergence with high discount factors."""
import pytest
import torch
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation import simulate_panel


class TestNFXPConvergence:
    """Tests for NFXP convergence with Rust bus data."""

    @pytest.fixture
    def rust_env(self):
        """Standard Rust bus environment."""
        return RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.9999,
        )

    def test_nfxp_recovers_true_parameters(self, rust_env):
        """NFXP should recover ground truth parameters within 3 SEs."""
        panel = simulate_panel(rust_env, n_individuals=300, n_periods=100, seed=42)
        utility = LinearUtility.from_environment(rust_env)

        # Use sufficient inner iterations for high discount
        estimator = NFXPEstimator(
            se_method="asymptotic",
            inner_max_iter=300000,  # Critical for β=0.9999
            inner_tol=1e-10,
            verbose=False,
        )

        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        true_params = rust_env.get_true_parameter_vector()

        # Check recovery within 3 standard errors
        for i, name in enumerate(result.parameter_names):
            error = abs(result.parameters[i].item() - true_params[i].item())
            se = result.standard_errors[i].item()
            assert error < 3 * se, (
                f"Parameter {name}: estimate={result.parameters[i].item():.6f}, "
                f"true={true_params[i].item():.6f}, error={error:.6f}, 3*SE={3*se:.6f}"
            )

    def test_inner_loop_converges_with_high_discount(self, rust_env):
        """Inner loop should converge without warnings when given enough iterations."""
        panel = simulate_panel(rust_env, n_individuals=50, n_periods=20, seed=42)
        utility = LinearUtility.from_environment(rust_env)

        # Default inner_max_iter is too low for β=0.9999
        estimator = NFXPEstimator(
            se_method="asymptotic",
            inner_max_iter=300000,
            inner_tol=1e-10,
            verbose=False,
        )

        # This should not print "Inner loop did not converge" warnings
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        assert result.converged
```

**Step 2: Run test to verify current failure**

Run: `pytest tests/test_nfxp_convergence.py::TestNFXPConvergence::test_nfxp_recovers_true_parameters -v`
Expected: FAIL - parameters don't match ground truth

**Step 3: Update NFXPEstimator default inner_max_iter**

In `src/econirl/estimation/nfxp.py`, find the `__init__` method and change default:

```python
# Before:
inner_max_iter: int = 1000

# After:
inner_max_iter: int = 100000  # Sufficient for β=0.9999
```

Also add a warning when discount is high:

```python
def estimate(self, ...):
    # Add at start of method:
    beta = problem.discount_factor
    if beta > 0.99 and self._inner_max_iter < 50000:
        import warnings
        warnings.warn(
            f"High discount factor β={beta} may require inner_max_iter > 50000. "
            f"Current: {self._inner_max_iter}. Consider increasing for convergence."
        )
```

**Step 4: Run test to verify fix**

Run: `pytest tests/test_nfxp_convergence.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_nfxp_convergence.py src/econirl/estimation/nfxp.py
git commit -m "fix: increase NFXP inner_max_iter default for high discount factors"
```

---

### Task 2: Add Action-Dependent Feature Support to MCE IRL

**Files:**
- Create: `src/econirl/preferences/action_reward.py`
- Test: `tests/test_action_reward.py`

**Step 1: Write the failing test**

```python
# tests/test_action_reward.py
"""Tests for action-dependent reward functions."""
import pytest
import torch
from econirl.preferences.action_reward import ActionDependentReward


class TestActionDependentReward:
    """Tests for ActionDependentReward class."""

    def test_init(self):
        """ActionDependentReward initializes correctly."""
        n_states = 90
        n_actions = 2
        n_features = 2

        # Feature matrix: (states, actions, features)
        features = torch.zeros((n_states, n_actions, n_features))
        for s in range(n_states):
            features[s, 0, 0] = -s  # Keep: operating cost
            features[s, 1, 1] = -1  # Replace: fixed cost

        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["theta_c", "RC"],
        )

        assert reward.num_states == n_states
        assert reward.num_actions == n_actions
        assert reward.num_parameters == n_features

    def test_compute_rust_utility(self):
        """Compute action-dependent reward for Rust model."""
        n_states = 90
        features = torch.zeros((n_states, 2, 2))
        for s in range(n_states):
            features[s, 0, 0] = -s
            features[s, 1, 1] = -1

        reward = ActionDependentReward(
            feature_matrix=features,
            parameter_names=["theta_c", "RC"],
        )

        params = torch.tensor([0.001, 3.0])
        R = reward.compute(params)

        assert R.shape == (n_states, 2)

        # Check values match Rust model
        assert torch.isclose(R[0, 0], torch.tensor(0.0), atol=1e-6)  # U(0, keep)
        assert torch.isclose(R[0, 1], torch.tensor(-3.0), atol=1e-6)  # U(0, replace)
        assert torch.isclose(R[5, 0], torch.tensor(-0.005), atol=1e-6)  # U(5, keep)
        assert torch.isclose(R[5, 1], torch.tensor(-3.0), atol=1e-6)  # U(5, replace)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_action_reward.py -v`
Expected: FAIL - module not found

**Step 3: Write ActionDependentReward implementation**

```python
# src/econirl/preferences/action_reward.py
"""Action-dependent reward function for IRL.

This extends the standard IRL reward to handle action-dependent features,
as required for structural models like Rust (1987) bus engine replacement.
"""
from __future__ import annotations

import torch
from econirl.preferences.base import BaseUtilityFunction


class ActionDependentReward(BaseUtilityFunction):
    """Action-dependent reward for IRL: R(s,a) = theta * phi(s,a).

    Unlike LinearReward (state-only), this supports different features
    per action, enabling Rust-style utility:
        U(s, keep) = -theta_c * s
        U(s, replace) = -RC

    Parameters
    ----------
    feature_matrix : torch.Tensor
        Shape (num_states, num_actions, num_features).
    parameter_names : list[str]
        Names for each parameter.
    """

    def __init__(
        self,
        feature_matrix: torch.Tensor,
        parameter_names: list[str],
    ):
        if feature_matrix.ndim != 3:
            raise ValueError(
                f"feature_matrix must be 3D (states, actions, features), "
                f"got shape {feature_matrix.shape}"
            )

        num_states, num_actions, num_features = feature_matrix.shape

        if len(parameter_names) != num_features:
            raise ValueError(
                f"parameter_names length {len(parameter_names)} != "
                f"num_features {num_features}"
            )

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            parameter_names=parameter_names,
            anchor_action=None,
        )

        self._feature_matrix = feature_matrix

    def compute(self, params: torch.Tensor) -> torch.Tensor:
        """Compute reward matrix R(s,a).

        Args:
            params: Parameter vector of shape (num_features,).

        Returns:
            Reward matrix of shape (num_states, num_actions).
        """
        self.validate_parameters(params)
        # R[s, a] = sum_k params[k] * features[s, a, k]
        return torch.einsum("sak,k->sa", self._feature_matrix, params)

    def compute_gradient(self, params: torch.Tensor) -> torch.Tensor:
        """Compute gradient dR/dtheta.

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_features).
        """
        # Gradient is just the feature matrix
        return self._feature_matrix

    @classmethod
    def from_rust_environment(cls, env) -> "ActionDependentReward":
        """Create from RustBusEnvironment.

        Args:
            env: RustBusEnvironment instance.

        Returns:
            ActionDependentReward with correct feature structure.
        """
        return cls(
            feature_matrix=env.feature_matrix,
            parameter_names=env.parameter_names,
        )
```

**Step 4: Run test**

Run: `pytest tests/test_action_reward.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/preferences/action_reward.py tests/test_action_reward.py
git commit -m "feat: add ActionDependentReward for action-specific features in IRL"
```

---

### Task 3: Update MCE IRL to Support Action-Dependent Features

**Files:**
- Modify: `src/econirl/estimation/mce_irl.py`
- Test: `tests/test_mce_irl_action_features.py`

**Step 1: Write test for MCE IRL with action-dependent features**

```python
# tests/test_mce_irl_action_features.py
"""Test MCE IRL with action-dependent features."""
import pytest
import torch
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.simulation import simulate_panel


class TestMCEIRLActionFeatures:
    """Tests for MCE IRL with action-dependent features."""

    @pytest.fixture
    def rust_env(self):
        return RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.99,  # Lower for faster convergence
        )

    def test_mce_irl_recovers_rust_parameters(self, rust_env):
        """MCE IRL with action features should recover Rust parameters."""
        panel = simulate_panel(rust_env, n_individuals=200, n_periods=50, seed=42)

        # Use action-dependent reward (same as NFXP utility)
        reward = ActionDependentReward.from_rust_environment(rust_env)

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=50000,
            outer_max_iter=500,
            learning_rate=0.1,
            se_method="hessian",
            compute_se=True,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        true_params = rust_env.get_true_parameter_vector()

        # MCE IRL recovers parameters up to a scale factor
        # Check that ratios are correct: theta_c/RC should match
        estimated_ratio = result.parameters[0] / result.parameters[1]
        true_ratio = true_params[0] / true_params[1]

        assert abs(estimated_ratio - true_ratio) / abs(true_ratio) < 0.1, (
            f"Ratio mismatch: estimated={estimated_ratio:.6f}, true={true_ratio:.6f}"
        )
```

**Step 2: Run test**

Run: `pytest tests/test_mce_irl_action_features.py -v`
Expected: Should pass if MCE IRL handles action-dependent features

**Step 3: Verify MCE IRL handles 3D feature matrix**

Check `src/econirl/estimation/mce_irl.py` - the `_compute_expected_features` method:

```python
def _compute_expected_features(self, ...):
    # Current implementation expects state-only features
    # Need to update for action-dependent features

    # If features are (states, actions, features), need different computation:
    if self._feature_matrix.ndim == 3:
        # Action-dependent: E[phi] = sum_s sum_a D(s) * pi(a|s) * phi(s,a)
        weighted_features = state_visitation.unsqueeze(1).unsqueeze(2) * \
                           policy.unsqueeze(2) * \
                           self._feature_matrix
        return weighted_features.sum(dim=(0, 1))
    else:
        # State-only: E[phi] = sum_s D(s) * phi(s)
        return torch.einsum("s,sf->f", state_visitation, self._feature_matrix)
```

**Step 4: Update MCE IRL for action-dependent features**

In `src/econirl/estimation/mce_irl.py`, update `_compute_expected_features`:

```python
def _compute_expected_features(
    self,
    policy: torch.Tensor,
    state_visitation: torch.Tensor,
    feature_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute expected features under policy.

    Handles both state-only and action-dependent features.
    """
    if feature_matrix.ndim == 3:
        # Action-dependent: (states, actions, features)
        # E[phi] = sum_s sum_a D(s) * pi(a|s) * phi(s,a,k)
        return torch.einsum("s,sa,sak->k", state_visitation, policy, feature_matrix)
    else:
        # State-only: (states, features)
        # E[phi] = sum_s D(s) * phi(s,k)
        return torch.einsum("s,sk->k", state_visitation, feature_matrix)
```

Also update `_compute_empirical_features`:

```python
def _compute_empirical_features(
    self,
    panel: Panel,
    feature_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute empirical feature expectations from data."""
    n_obs = sum(len(t) for t in panel.trajectories)

    if feature_matrix.ndim == 3:
        # Action-dependent features
        feature_sum = torch.zeros(feature_matrix.shape[2])
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                feature_sum += feature_matrix[s, a, :]
    else:
        # State-only features
        feature_sum = torch.zeros(feature_matrix.shape[1])
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                feature_sum += feature_matrix[s, :]

    return feature_sum / n_obs
```

**Step 5: Run tests**

Run: `pytest tests/test_mce_irl_action_features.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/econirl/estimation/mce_irl.py tests/test_mce_irl_action_features.py
git commit -m "feat: support action-dependent features in MCE IRL"
```

---

### Task 4: Create Comparison Notebook with Fixed Estimators

**Files:**
- Update: `examples/nfxp_vs_mceirl_comparison.ipynb`

**Step 1: Update notebook to use correct settings**

```python
# Key changes:
# 1. NFXP: inner_max_iter=300000 for β=0.9999
# 2. MCE IRL: Use ActionDependentReward instead of LinearReward
# 3. Lower discount (β=0.99) for faster demo

# NFXP setup
nfxp = NFXPEstimator(
    se_method="asymptotic",
    inner_max_iter=300000,  # Critical!
    inner_tol=1e-10,
    verbose=True,
)

# MCE IRL setup
from econirl.preferences.action_reward import ActionDependentReward
reward = ActionDependentReward.from_rust_environment(env)

config = MCEIRLConfig(
    verbose=True,
    inner_max_iter=50000,
    outer_max_iter=500,
    learning_rate=0.1,
    se_method="hessian",
)
```

**Step 2: Run notebook and verify parameter recovery**

Run: `jupyter nbconvert --execute examples/nfxp_vs_mceirl_comparison.ipynb`
Expected: Both NFXP and MCE IRL recover parameters close to ground truth

**Step 3: Commit**

```bash
git add examples/nfxp_vs_mceirl_comparison.ipynb
git commit -m "docs: update comparison notebook with fixed estimator settings"
```

---

### Task 5: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v --ignore=tests/benchmarks`
Expected: All tests pass

**Step 2: Final commit**

```bash
git add -A
git commit -m "fix: complete parameter recovery fixes for NFXP and MCE IRL"
```

---

## Summary

The parameter recovery failures had two root causes:

1. **NFXP**: Inner loop (value iteration) never converged because β=0.9999 requires ~276,000 iterations but default was 1,000. Fix: Increase `inner_max_iter` default and add warning.

2. **MCE IRL**: Used state-only features (LinearReward) but Rust model has action-dependent utilities. Fix: Create ActionDependentReward class that handles (states, actions, features) feature matrices.

After these fixes, both estimators should recover ground truth parameters.
