# MCE IRL with Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Maximum Causal Entropy IRL with proper inference (standard errors, confidence intervals) and prediction capabilities for the bus engine replacement problem.

**Architecture:** MCE IRL uses soft value iteration (backward pass) and state visitation frequency computation (forward pass) to match feature expectations. We'll support action-dependent utilities for structural estimation problems like Rust's bus engine model, with numerical Hessian for inference.

**Tech Stack:** PyTorch, SciPy (optimize), NumPy, existing econirl infrastructure (DDCProblem, Panel, SoftBellmanOperator)

---

### Task 1: Fix MCE IRL Core Estimator

**Files:**
- Modify: `src/econirl/estimation/mce_irl.py:1-550`
- Test: `tests/test_mce_irl_core.py`

**Step 1: Write the failing test for MCE IRL convergence**

```python
# tests/test_mce_irl_core.py
"""Tests for MCE IRL core estimator."""
import pytest
import torch
import numpy as np

from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.reward import LinearReward


class TestMCEIRLConvergence:
    """Test that MCE IRL converges on simple problems."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple 10-state MDP with known structure."""
        n_states = 10
        problem = DDCProblem(
            num_states=n_states,
            num_actions=2,
            discount_factor=0.95,
        )

        # Deterministic transitions: keep -> next state, replace -> state 0
        transitions = torch.zeros((2, n_states, n_states))
        for s in range(n_states):
            transitions[0, s, min(s + 1, n_states - 1)] = 1.0  # keep
            transitions[1, s, 0] = 1.0  # replace

        return problem, transitions

    @pytest.fixture
    def synthetic_panel(self, simple_problem):
        """Generate synthetic data from a known policy."""
        problem, transitions = simple_problem
        n_states = problem.num_states

        np.random.seed(42)
        trajectories = []

        for i in range(20):
            states, actions = [], []
            s = 0
            for t in range(50):
                states.append(s)
                # Replace with higher prob at high states
                p_replace = 0.05 + 0.15 * s / n_states
                a = 1 if np.random.random() < p_replace else 0
                actions.append(a)
                s = 0 if a == 1 else min(s + 1, n_states - 1)

            traj = Trajectory(
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor([0] * 50, dtype=torch.long),
                individual_id=i,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def test_mce_irl_converges(self, simple_problem, synthetic_panel):
        """MCE IRL should converge on simple problem."""
        problem, transitions = simple_problem

        # State features
        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=100,
            learning_rate=0.5,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        assert result.converged or result.num_iterations < config.outer_max_iter
        assert result.log_likelihood > -float("inf")
        assert result.policy is not None
        assert result.policy.shape == (10, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mce_irl_core.py::TestMCEIRLConvergence::test_mce_irl_converges -v`
Expected: PASS (test should pass with current implementation)

**Step 3: Write test for feature matching**

```python
    def test_feature_matching(self, simple_problem, synthetic_panel):
        """MCE IRL should approximately match empirical feature expectations."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=False,
            inner_max_iter=500,
            outer_max_iter=200,
            learning_rate=0.5,
            outer_tol=1e-4,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        result = estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

        # Check feature difference is small
        feature_diff = result.metadata.get("feature_difference", float("inf"))
        assert feature_diff < 0.01, f"Feature difference {feature_diff} too large"
```

**Step 4: Run test**

Run: `pytest tests/test_mce_irl_core.py::TestMCEIRLConvergence::test_feature_matching -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_mce_irl_core.py
git commit -m "test: add MCE IRL convergence and feature matching tests"
```

---

### Task 2: Add Action-Dependent Utility Support

**Files:**
- Create: `src/econirl/preferences/action_utility.py`
- Test: `tests/test_action_utility.py`

**Step 1: Write the failing test**

```python
# tests/test_action_utility.py
"""Tests for action-dependent utility functions."""
import pytest
import torch

from econirl.preferences.action_utility import ActionDependentUtility


class TestActionDependentUtility:
    """Tests for ActionDependentUtility class."""

    def test_init(self):
        """ActionDependentUtility initializes correctly."""
        n_states = 90
        n_actions = 2

        utility = ActionDependentUtility(
            num_states=n_states,
            num_actions=n_actions,
            parameter_names=["theta_c", "RC"],
        )

        assert utility.num_states == n_states
        assert utility.num_actions == n_actions
        assert utility.num_parameters == 2
        assert utility.parameter_names == ["theta_c", "RC"]

    def test_compute_rust_bus_utility(self):
        """Compute utility for Rust bus engine model."""
        n_states = 90

        utility = ActionDependentUtility(
            num_states=n_states,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
        )

        # Rust model: U(s, keep) = -theta_c * s, U(s, replace) = -RC
        params = torch.tensor([0.001, 3.0])  # theta_c, RC
        U = utility.compute(params)

        assert U.shape == (n_states, 2)

        # Check values
        assert torch.isclose(U[0, 0], torch.tensor(0.0), atol=1e-6)  # U(0, keep) = 0
        assert torch.isclose(U[0, 1], torch.tensor(-3.0), atol=1e-6)  # U(0, replace) = -RC
        assert torch.isclose(U[50, 0], torch.tensor(-0.05), atol=1e-6)  # U(50, keep) = -0.001*50
        assert torch.isclose(U[50, 1], torch.tensor(-3.0), atol=1e-6)  # U(50, replace) = -RC
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_action_utility.py::TestActionDependentUtility -v`
Expected: FAIL with "No module named 'econirl.preferences.action_utility'"

**Step 3: Write the implementation**

```python
# src/econirl/preferences/action_utility.py
"""Action-dependent utility specification for structural estimation.

This module implements utility functions that depend on both state and action,
as required for structural models like Rust (1987) bus engine replacement.

The utility structure is:
    U(s, a=keep) = -theta_c * s / scale   (operating cost)
    U(s, a=replace) = -RC                  (replacement cost)

This differs from IRL reward functions which are typically state-only.
"""

from __future__ import annotations

import torch

from econirl.preferences.base import BaseUtilityFunction


class ActionDependentUtility(BaseUtilityFunction):
    """Action-dependent utility for structural DDC models.

    Implements the Rust (1987) utility structure:
        U(s, a=0) = -theta_c * s / scale  (keep: operating cost)
        U(s, a=1) = -RC                    (replace: fixed cost)

    Parameters
    ----------
    num_states : int
        Number of discrete states.
    num_actions : int
        Number of actions (typically 2: keep, replace).
    parameter_names : list[str]
        Names for parameters, e.g., ["theta_c", "RC"].
    scale : float, default=100.0
        Scale factor for state (divides state index).

    Examples
    --------
    >>> utility = ActionDependentUtility(
    ...     num_states=90,
    ...     num_actions=2,
    ...     parameter_names=["theta_c", "RC"],
    ... )
    >>> params = torch.tensor([0.001, 3.0])
    >>> U = utility.compute(params)
    >>> U.shape
    torch.Size([90, 2])
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int = 2,
        parameter_names: list[str] | None = None,
        scale: float = 100.0,
    ):
        if parameter_names is None:
            parameter_names = ["theta_c", "RC"]

        if len(parameter_names) != 2:
            raise ValueError(
                f"ActionDependentUtility requires exactly 2 parameters, "
                f"got {len(parameter_names)}"
            )

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            parameter_names=parameter_names,
            anchor_action=None,
        )

        self._scale = scale
        self._state_indices = torch.arange(num_states, dtype=torch.float32)

    def compute(self, params: torch.Tensor) -> torch.Tensor:
        """Compute utility matrix U(s, a).

        Args:
            params: Parameter vector [theta_c, RC].

        Returns:
            Utility matrix of shape (num_states, num_actions).
        """
        self.validate_parameters(params)

        theta_c = params[0]
        RC = params[1]

        U = torch.zeros((self.num_states, self.num_actions), dtype=params.dtype)

        # U(s, a=0) = -theta_c * s / scale (keep)
        U[:, 0] = -theta_c * self._state_indices / self._scale

        # U(s, a=1) = -RC (replace)
        U[:, 1] = -RC

        return U

    def compute_gradient(self, params: torch.Tensor) -> torch.Tensor:
        """Compute gradient of utility w.r.t. parameters.

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_parameters).
        """
        grad = torch.zeros(
            (self.num_states, self.num_actions, self.num_parameters),
            dtype=params.dtype,
        )

        # dU/d(theta_c) for action 0 (keep)
        grad[:, 0, 0] = -self._state_indices / self._scale

        # dU/d(RC) for action 1 (replace)
        grad[:, 1, 1] = -1.0

        return grad

    def get_initial_parameters(self) -> torch.Tensor:
        """Return reasonable initial parameter values."""
        return torch.tensor([0.001, 3.0], dtype=torch.float32)

    def get_parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounds: theta_c >= 0, RC >= 0."""
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([float("inf"), float("inf")])
        return lower, upper

    def __repr__(self) -> str:
        return (
            f"ActionDependentUtility("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"parameters={self.parameter_names})"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_action_utility.py::TestActionDependentUtility -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/preferences/action_utility.py tests/test_action_utility.py
git commit -m "feat: add ActionDependentUtility for structural DDC models"
```

---

### Task 3: Implement Numerical Hessian for Inference

**Files:**
- Modify: `src/econirl/estimation/mce_irl.py`
- Test: `tests/test_mce_irl_core.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mce_irl_core.py
class TestMCEIRLInference:
    """Tests for MCE IRL inference capabilities."""

    @pytest.fixture
    def fitted_result(self, simple_problem, synthetic_panel):
        """Fit MCE IRL and return result."""
        problem, transitions = simple_problem

        features = torch.arange(10, dtype=torch.float32).unsqueeze(1) / 10
        reward_fn = LinearReward(
            state_features=features,
            parameter_names=["cost"],
            n_actions=2,
        )

        config = MCEIRLConfig(
            compute_se=True,
            se_method="hessian",
            inner_max_iter=500,
            outer_max_iter=100,
            learning_rate=0.5,
            verbose=False,
        )
        estimator = MCEIRLEstimator(config=config)

        return estimator.estimate(
            panel=synthetic_panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
        )

    def test_standard_errors_computed(self, fitted_result):
        """Standard errors should be computed when requested."""
        # Standard errors should be finite
        se = fitted_result.standard_errors
        assert se is not None
        assert len(se) == 1  # One parameter
        assert torch.isfinite(se).all(), f"SE not finite: {se}"

    def test_confidence_interval_valid(self, fitted_result):
        """95% CI should be valid."""
        params = fitted_result.parameters
        se = fitted_result.standard_errors

        ci_low = params - 1.96 * se
        ci_high = params + 1.96 * se

        # CI should contain the point estimate
        assert (ci_low <= params).all()
        assert (params <= ci_high).all()
```

**Step 2: Run test to verify current state**

Run: `pytest tests/test_mce_irl_core.py::TestMCEIRLInference -v`
Expected: May fail if Hessian computation has issues

**Step 3: Fix the numerical Hessian computation**

Update `_numerical_hessian` in `src/econirl/estimation/mce_irl.py`:

```python
    def _numerical_hessian(
        self,
        params: torch.Tensor,
        panel: Panel,
        reward_fn: LinearReward,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_dist: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Compute numerical Hessian of the log-likelihood.

        Uses central differences for better numerical accuracy.
        """
        operator = SoftBellmanOperator(problem, transitions)
        n_params = len(params)
        hessian = torch.zeros((n_params, n_params))

        def ll_at(p):
            """Compute log-likelihood at parameter value p."""
            reward_matrix = reward_fn.compute(p)
            V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)

            ll = 0.0
            for traj in panel.trajectories:
                for t in range(len(traj)):
                    state = traj.states[t].item()
                    action = traj.actions[t].item()
                    ll += log_probs[state, action].item()
            return ll

        # Compute Hessian using central differences
        for i in range(n_params):
            for j in range(i, n_params):
                p_pp = params.clone()
                p_pp[i] += eps
                p_pp[j] += eps

                p_pm = params.clone()
                p_pm[i] += eps
                p_pm[j] -= eps

                p_mp = params.clone()
                p_mp[i] -= eps
                p_mp[j] += eps

                p_mm = params.clone()
                p_mm[i] -= eps
                p_mm[j] -= eps

                # Central difference formula for second derivative
                h_ij = (ll_at(p_pp) - ll_at(p_pm) - ll_at(p_mp) + ll_at(p_mm)) / (4 * eps * eps)

                hessian[i, j] = h_ij
                hessian[j, i] = h_ij

        return hessian
```

**Step 4: Run test**

Run: `pytest tests/test_mce_irl_core.py::TestMCEIRLInference -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/estimation/mce_irl.py tests/test_mce_irl_core.py
git commit -m "fix: improve numerical Hessian computation for inference"
```

---

### Task 4: Add Sklearn-Style MCEIRL Wrapper Tests

**Files:**
- Modify: `src/econirl/estimators/mce_irl.py`
- Test: `tests/test_mce_irl_sklearn.py`

**Step 1: Write the failing test**

```python
# tests/test_mce_irl_sklearn.py
"""Tests for sklearn-style MCEIRL estimator."""
import pytest
import numpy as np
import pandas as pd
import torch

from econirl.estimators.mce_irl import MCEIRL


class TestMCEIRLSklearn:
    """Tests for sklearn-style MCEIRL."""

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
                next_state = 0 if action == 1 else min(state + np.random.choice([0, 1, 2]), n_states - 1)
                data.append({
                    "id": i,
                    "period": t,
                    "state": state,
                    "action": action,
                })
                state = next_state

        return pd.DataFrame(data)

    def test_fit_returns_self(self, sample_df):
        """fit() should return self for method chaining."""
        model = MCEIRL(n_states=20, discount=0.95, verbose=False)
        result = model.fit(sample_df, state="state", action="action", id="id")
        assert result is model

    def test_params_after_fit(self, sample_df):
        """params_ should be populated after fit."""
        features = np.arange(20).reshape(-1, 1) / 20
        model = MCEIRL(
            n_states=20,
            discount=0.95,
            feature_matrix=features,
            feature_names=["cost"],
            se_method="hessian",
            verbose=False,
        )
        model.fit(sample_df, state="state", action="action", id="id")

        assert model.params_ is not None
        assert "cost" in model.params_
        assert model.coef_ is not None

    def test_predict_proba(self, sample_df):
        """predict_proba should return valid probabilities."""
        model = MCEIRL(n_states=20, discount=0.95, verbose=False)
        model.fit(sample_df, state="state", action="action", id="id")

        proba = model.predict_proba(np.array([0, 5, 10, 15, 19]))

        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_summary(self, sample_df):
        """summary() should return formatted string."""
        model = MCEIRL(n_states=20, discount=0.95, verbose=False)
        model.fit(sample_df, state="state", action="action", id="id")

        summary = model.summary()

        assert isinstance(summary, str)
        assert "MCE IRL" in summary or "Maximum Causal Entropy" in summary
```

**Step 2: Run test**

Run: `pytest tests/test_mce_irl_sklearn.py::TestMCEIRLSklearn -v`
Expected: Most tests should pass

**Step 3: Fix any failing tests by updating MCEIRL class**

If tests fail, update `src/econirl/estimators/mce_irl.py` accordingly.

**Step 4: Run all tests**

Run: `pytest tests/test_mce_irl_sklearn.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/estimators/mce_irl.py tests/test_mce_irl_sklearn.py
git commit -m "test: add sklearn-style MCEIRL wrapper tests"
```

---

### Task 5: Integration Test with Rust Bus Data

**Files:**
- Test: `tests/integration/test_mce_irl_bus.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_mce_irl_bus.py
"""Integration test: MCE IRL on Rust bus engine data."""
import pytest
import numpy as np
import torch

from econirl.datasets import load_rust_bus
from econirl.estimators.mce_irl import MCEIRL


class TestMCEIRLBusIntegration:
    """Integration tests for MCE IRL on bus engine data."""

    @pytest.fixture
    def bus_data(self):
        """Load bus data."""
        return load_rust_bus()

    def test_mce_irl_fits_bus_data(self, bus_data):
        """MCE IRL should fit bus data without error."""
        df = bus_data

        n_states = 90
        features = np.arange(n_states).reshape(-1, 1) / 100

        model = MCEIRL(
            n_states=n_states,
            n_actions=2,
            discount=0.99,
            feature_matrix=features,
            feature_names=["mileage"],
            se_method="hessian",
            inner_max_iter=2000,
            verbose=False,
        )

        model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        assert model.params_ is not None
        assert model.log_likelihood_ is not None
        assert model.policy_ is not None

    def test_replacement_probability_reasonable(self, bus_data):
        """Predicted replacement probability should be in reasonable range."""
        df = bus_data
        n_states = 90
        features = np.arange(n_states).reshape(-1, 1) / 100

        model = MCEIRL(
            n_states=n_states,
            discount=0.99,
            feature_matrix=features,
            feature_names=["mileage"],
            verbose=False,
        )
        model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        # Get replacement probability at state 0
        proba = model.predict_proba(np.array([0]))
        p_replace = proba[0, 1]

        # Should be in reasonable range (empirical is ~5%)
        assert 0.01 < p_replace < 0.5, f"P(replace|s=0) = {p_replace} out of range"

    def test_log_likelihood_improves_over_baseline(self, bus_data):
        """Log-likelihood should be better than uniform random policy."""
        df = bus_data
        n_states = 90
        features = np.arange(n_states).reshape(-1, 1) / 100

        model = MCEIRL(
            n_states=n_states,
            discount=0.99,
            feature_matrix=features,
            feature_names=["mileage"],
            verbose=False,
        )
        model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        # Uniform random policy LL: n_obs * log(0.5)
        n_obs = len(df)
        uniform_ll = n_obs * np.log(0.5)

        assert model.log_likelihood_ > uniform_ll, (
            f"Model LL ({model.log_likelihood_:.2f}) not better than "
            f"uniform ({uniform_ll:.2f})"
        )
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_mce_irl_bus.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_mce_irl_bus.py
git commit -m "test: add MCE IRL integration test with bus data"
```

---

### Task 6: Export MCE IRL from Package

**Files:**
- Modify: `src/econirl/__init__.py`
- Modify: `src/econirl/estimators/__init__.py`
- Modify: `src/econirl/estimation/__init__.py`

**Step 1: Update estimators __init__.py**

```python
# Add to src/econirl/estimators/__init__.py
from econirl.estimators.mce_irl import MCEIRL

__all__ = [
    # ... existing exports ...
    "MCEIRL",
]
```

**Step 2: Update estimation __init__.py**

```python
# Add to src/econirl/estimation/__init__.py
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig

__all__ = [
    # ... existing exports ...
    "MCEIRLEstimator",
    "MCEIRLConfig",
]
```

**Step 3: Update main __init__.py**

```python
# Add to src/econirl/__init__.py
from econirl.estimators.mce_irl import MCEIRL

__all__ = [
    # ... existing exports ...
    "MCEIRL",
]
```

**Step 4: Test imports**

Run: `python3 -c "from econirl import MCEIRL; print('Import OK')"`
Expected: "Import OK"

**Step 5: Commit**

```bash
git add src/econirl/__init__.py src/econirl/estimators/__init__.py src/econirl/estimation/__init__.py
git commit -m "feat: export MCEIRL from package"
```

---

### Task 7: Create Example Notebook

**Files:**
- Create: `examples/mce_irl_bus_example.py`

**Step 1: Write the example script**

```python
# examples/mce_irl_bus_example.py
"""
Example: Maximum Causal Entropy IRL on Bus Engine Data

This example demonstrates how to:
1. Load the Rust bus engine replacement data
2. Fit MCE IRL to recover reward parameters
3. Compute standard errors and confidence intervals
4. Make predictions using the learned model
"""

import numpy as np
from econirl.datasets import load_rust_bus
from econirl.estimators import MCEIRL

# Load data
print("Loading Rust Bus Engine Data")
print("=" * 50)
df = load_rust_bus()
print(f"Observations: {len(df):,}")
print(f"Buses: {df['bus_id'].nunique()}")
print(f"Replacement rate: {df['replaced'].mean():.2%}")
print()

# Define features
n_states = 90
features = np.arange(n_states).reshape(-1, 1) / 100  # Normalized mileage

# Fit model
print("Fitting MCE IRL")
print("=" * 50)
model = MCEIRL(
    n_states=n_states,
    n_actions=2,
    discount=0.99,
    feature_matrix=features,
    feature_names=["mileage_cost"],
    se_method="hessian",
    verbose=True,
)

model.fit(
    data=df,
    state="mileage_bin",
    action="replaced",
    id="bus_id",
)

# Results
print()
print(model.summary())

# Predictions
print()
print("Predictions")
print("=" * 50)
test_states = np.array([0, 20, 40, 60, 80])
proba = model.predict_proba(test_states)
print(f"{'State':>6} {'P(keep)':>10} {'P(replace)':>12}")
for i, s in enumerate(test_states):
    print(f"{s:>6} {proba[i, 0]:>10.4f} {proba[i, 1]:>12.4f}")

print()
print("Done!")
```

**Step 2: Run the example**

Run: `python3 examples/mce_irl_bus_example.py`
Expected: Script runs without error, prints results

**Step 3: Commit**

```bash
git add examples/mce_irl_bus_example.py
git commit -m "docs: add MCE IRL bus engine example"
```

---

### Task 8: Run Full Test Suite

**Step 1: Run all MCE IRL tests**

Run: `pytest tests/test_mce_irl*.py tests/integration/test_mce_irl*.py -v`
Expected: All tests pass

**Step 2: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/benchmarks`
Expected: All tests pass (or only unrelated failures)

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete MCE IRL implementation with inference"
```

---

## Summary

This plan implements:
1. **MCE IRL Core** - Soft value iteration + state visitation frequency matching
2. **Action-Dependent Utility** - For structural models like Rust (1987)
3. **Inference** - Numerical Hessian for standard errors
4. **Sklearn API** - MCEIRL class with fit/predict/summary
5. **Tests** - Unit tests, integration tests with bus data
6. **Example** - Working example script
