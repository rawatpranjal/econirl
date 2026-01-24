# sklearn-Style API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor econirl to use scikit-learn style API: `Estimator(config).fit(data)` with results as properties.

**Architecture:** Create new estimator classes (NFXP, CCP, NPL) that wrap existing implementation. Accept DataFrame input with column names. Store fitted results as properties with trailing underscore. Keep backward compatibility by retaining old classes.

**Tech Stack:** Python, pandas, torch, scipy

---

## Task 1: Create Utility Base Class with Built-in Utilities

**Files:**
- Create: `src/econirl/utilities.py`
- Test: `tests/test_utilities.py`

**Step 1: Write the failing test**

```python
# tests/test_utilities.py
import pytest
import torch
from econirl.utilities import Utility, LinearCost


def test_linear_cost_n_params():
    """LinearCost utility has 2 parameters: theta_c and RC."""
    util = LinearCost()
    assert util.n_params == 2


def test_linear_cost_param_names():
    """LinearCost has named parameters."""
    util = LinearCost()
    assert util.param_names == ["theta_c", "RC"]


def test_linear_cost_call():
    """LinearCost computes u = -theta_c * s * (1-a) - RC * a."""
    util = LinearCost()
    params = torch.tensor([0.001, 10.0])

    # State 50, action 0 (keep): u = -0.001 * 50 * 1 = -0.05
    u_keep = util(state=50, action=0, params=params)
    assert torch.isclose(u_keep, torch.tensor(-0.05), atol=1e-6)

    # State 50, action 1 (replace): u = -10.0
    u_replace = util(state=50, action=1, params=params)
    assert torch.isclose(u_replace, torch.tensor(-10.0), atol=1e-6)


def test_linear_cost_matrix():
    """LinearCost.matrix() returns (n_states, n_actions) utility matrix."""
    util = LinearCost()
    params = torch.tensor([0.001, 10.0])

    U = util.matrix(n_states=90, params=params)

    assert U.shape == (90, 2)
    # Check specific values
    assert torch.isclose(U[0, 0], torch.tensor(0.0), atol=1e-6)  # state 0, keep
    assert torch.isclose(U[0, 1], torch.tensor(-10.0), atol=1e-6)  # state 0, replace
    assert torch.isclose(U[50, 0], torch.tensor(-0.05), atol=1e-6)  # state 50, keep


def test_callable_utility():
    """Custom callable works as utility."""
    def my_utility(state, action, params):
        return -params[0] * state * (1 - action) - params[1] * action

    from econirl.utilities import make_utility
    util = make_utility(my_utility, n_params=2, param_names=["cost", "RC"])

    assert util.n_params == 2
    assert util.param_names == ["cost", "RC"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utilities.py -v`
Expected: FAIL with "No module named 'econirl.utilities'"

**Step 3: Write minimal implementation**

```python
# src/econirl/utilities.py
"""Utility function specifications for DDC models.

Provides a simple interface for specifying utility functions,
following scikit-learn conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


class Utility(ABC):
    """Base class for utility functions.

    A utility function maps (state, action, params) -> flow utility.

    Subclasses must implement:
        - n_params: number of parameters
        - param_names: list of parameter names
        - __call__: compute utility for single (state, action)
        - matrix: compute utility matrix for all (state, action) pairs
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters."""
        ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of parameters."""
        ...

    @property
    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        """Bounds for each parameter. Default: no bounds."""
        return [(None, None)] * self.n_params

    @property
    def param_init(self) -> list[float]:
        """Initial values for parameters. Default: zeros."""
        return [0.0] * self.n_params

    @abstractmethod
    def __call__(
        self, state: int | torch.Tensor, action: int | torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """Compute utility for given state, action, and parameters."""
        ...

    @abstractmethod
    def matrix(self, n_states: int, params: torch.Tensor) -> torch.Tensor:
        """Compute utility matrix of shape (n_states, n_actions)."""
        ...


class LinearCost(Utility):
    """Linear cost utility: u = -theta_c * s * (1-a) - RC * a

    This is the standard Rust (1987) utility specification:
    - theta_c: per-unit operating/maintenance cost
    - RC: replacement cost

    Action 0 = keep (incur operating cost)
    Action 1 = replace (incur replacement cost, reset state)
    """

    @property
    def n_params(self) -> int:
        return 2

    @property
    def param_names(self) -> list[str]:
        return ["theta_c", "RC"]

    @property
    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        return [(0, None), (0, None)]  # Both non-negative

    @property
    def param_init(self) -> list[float]:
        return [0.001, 5.0]

    def __call__(
        self, state: int | torch.Tensor, action: int | torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        theta_c, RC = params[0], params[1]
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        return -theta_c * state * (1 - action) - RC * action

    def matrix(self, n_states: int, params: torch.Tensor) -> torch.Tensor:
        theta_c, RC = params[0], params[1]
        states = torch.arange(n_states, dtype=torch.float32)

        # Shape: (n_states, 2)
        U = torch.zeros(n_states, 2)
        U[:, 0] = -theta_c * states  # Keep: -theta_c * s
        U[:, 1] = -RC                 # Replace: -RC
        return U


@dataclass
class CallableUtility(Utility):
    """Wrapper to use a callable as a Utility."""

    _fn: Callable
    _n_params: int
    _param_names: list[str]
    _param_bounds: list[tuple[float | None, float | None]] | None = None
    _param_init: list[float] | None = None

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @property
    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        if self._param_bounds is not None:
            return self._param_bounds
        return [(None, None)] * self.n_params

    @property
    def param_init(self) -> list[float]:
        if self._param_init is not None:
            return self._param_init
        return [0.0] * self.n_params

    def __call__(
        self, state: int | torch.Tensor, action: int | torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        return self._fn(state, action, params)

    def matrix(self, n_states: int, params: torch.Tensor) -> torch.Tensor:
        U = torch.zeros(n_states, 2)  # Assumes binary action
        for s in range(n_states):
            for a in range(2):
                U[s, a] = self._fn(s, a, params)
        return U


def make_utility(
    fn: Callable,
    n_params: int,
    param_names: list[str] | None = None,
    param_bounds: list[tuple[float | None, float | None]] | None = None,
    param_init: list[float] | None = None,
) -> Utility:
    """Create a Utility from a callable.

    Args:
        fn: Function (state, action, params) -> utility
        n_params: Number of parameters
        param_names: Names for parameters (default: param_0, param_1, ...)
        param_bounds: Bounds for parameters
        param_init: Initial values for parameters

    Returns:
        Utility object
    """
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    return CallableUtility(
        _fn=fn,
        _n_params=n_params,
        _param_names=param_names,
        _param_bounds=param_bounds,
        _param_init=param_init,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_utilities.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/utilities.py tests/test_utilities.py
git commit -m "feat: add sklearn-style Utility classes

Add Utility base class, LinearCost built-in, and make_utility factory.
These provide a simpler interface than the existing UtilityFunction."
```

---

## Task 2: Create TransitionEstimator Class

**Files:**
- Create: `src/econirl/transitions.py`
- Test: `tests/test_transitions_sklearn.py`

**Step 1: Write the failing test**

```python
# tests/test_transitions_sklearn.py
import pytest
import pandas as pd
import numpy as np
import torch
from econirl.transitions import TransitionEstimator


def test_transition_estimator_fit_returns_self():
    """fit() returns self for chaining."""
    df = pd.DataFrame({
        "bus_id": [1, 1, 1, 2, 2, 2],
        "mileage_bin": [0, 1, 2, 0, 2, 3],
        "replaced": [0, 0, 0, 0, 0, 0],
    })

    est = TransitionEstimator(n_states=90)
    result = est.fit(df, state="mileage_bin", id="bus_id")

    assert result is est


def test_transition_estimator_matrix_():
    """matrix_ is available after fit."""
    df = pd.DataFrame({
        "bus_id": [1, 1, 1, 1],
        "mileage_bin": [0, 1, 2, 3],
        "replaced": [0, 0, 0, 0],
    })

    est = TransitionEstimator(n_states=90)
    est.fit(df, state="mileage_bin", id="bus_id")

    assert hasattr(est, "matrix_")
    assert est.matrix_.shape == (90, 90)


def test_transition_estimator_probs_():
    """probs_ contains estimated (theta_0, theta_1, theta_2)."""
    df = pd.DataFrame({
        "bus_id": [1, 1, 1, 1, 1],
        "mileage_bin": [0, 1, 2, 3, 4],
        "replaced": [0, 0, 0, 0, 0],
    })

    est = TransitionEstimator(n_states=90)
    est.fit(df, state="mileage_bin", id="bus_id")

    assert hasattr(est, "probs_")
    assert len(est.probs_) == 3
    assert sum(est.probs_) == pytest.approx(1.0)


def test_transition_estimator_summary():
    """summary() returns string."""
    df = pd.DataFrame({
        "bus_id": [1, 1, 1, 1, 1],
        "mileage_bin": [0, 1, 2, 3, 4],
        "replaced": [0, 0, 0, 0, 0],
    })

    est = TransitionEstimator(n_states=90)
    est.fit(df, state="mileage_bin", id="bus_id")

    summary = est.summary()
    assert isinstance(summary, str)
    assert "theta_0" in summary.lower() or "θ" in summary
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_transitions_sklearn.py -v`
Expected: FAIL with "No module named 'econirl.transitions'"

**Step 3: Write minimal implementation**

```python
# src/econirl/transitions.py
"""Transition probability estimation for DDC models.

Provides sklearn-style TransitionEstimator class.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch


class TransitionEstimator:
    """Estimate transition probabilities from panel data.

    For the Rust bus model, estimates (theta_0, theta_1, theta_2):
    - theta_0: P(stay at same mileage bin)
    - theta_1: P(increase by 1 bin)
    - theta_2: P(increase by 2 bins)

    Example:
        >>> est = TransitionEstimator(n_states=90)
        >>> est.fit(df, state="mileage_bin", id="bus_id")
        >>> print(est.probs_)
        >>> print(est.matrix_.shape)
    """

    def __init__(self, n_states: int = 90, max_increase: int = 2):
        """Initialize the estimator.

        Args:
            n_states: Number of discrete states
            max_increase: Maximum state increase per period (default 2)
        """
        self.n_states = n_states
        self.max_increase = max_increase

        # Fitted attributes (set by fit)
        self.matrix_: torch.Tensor | None = None
        self.probs_: np.ndarray | None = None
        self.n_transitions_: int | None = None

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        id: str,
        action: str | None = None,
    ) -> "TransitionEstimator":
        """Fit transition probabilities from data.

        Args:
            data: DataFrame with panel data
            state: Column name for state variable
            id: Column name for individual identifier
            action: Column name for action (if provided, only count non-replacement transitions)

        Returns:
            self
        """
        # Count transitions
        counts = np.zeros(self.max_increase + 1)
        n_transitions = 0

        for ind_id in data[id].unique():
            ind_data = data[data[id] == ind_id].sort_index()
            states = ind_data[state].values

            if action is not None:
                actions = ind_data[action].values
            else:
                actions = np.zeros(len(states))

            for t in range(len(states) - 1):
                # Skip if action was replacement (state resets)
                if actions[t] == 1:
                    continue

                s_t = states[t]
                s_next = states[t + 1]

                # Compute increase (clamped to max_increase)
                increase = min(s_next - s_t, self.max_increase)
                increase = max(increase, 0)  # No negative increases

                counts[increase] += 1
                n_transitions += 1

        # Normalize to probabilities
        if n_transitions > 0:
            probs = counts / n_transitions
        else:
            probs = np.array([1.0, 0.0, 0.0])

        self.probs_ = probs
        self.n_transitions_ = n_transitions

        # Build transition matrix
        self.matrix_ = self._build_matrix(probs)

        return self

    def _build_matrix(self, probs: np.ndarray) -> torch.Tensor:
        """Build transition matrix from probabilities."""
        P = torch.zeros(self.n_states, self.n_states)

        for s in range(self.n_states):
            for j, p in enumerate(probs):
                next_s = min(s + j, self.n_states - 1)
                P[s, next_s] += p

        return P

    def summary(self) -> str:
        """Generate summary of estimated transitions."""
        if self.probs_ is None:
            return "TransitionEstimator not fitted. Call fit() first."

        lines = [
            "Transition Probability Estimates",
            "=" * 40,
            f"Number of transitions: {self.n_transitions_:,}",
            "",
            f"θ₀ (stay):    {self.probs_[0]:.4f}",
            f"θ₁ (+1 bin):  {self.probs_[1]:.4f}",
            f"θ₂ (+2 bins): {self.probs_[2]:.4f}",
            "=" * 40,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.probs_ is None:
            return f"TransitionEstimator(n_states={self.n_states}, fitted=False)"
        return f"TransitionEstimator(n_states={self.n_states}, probs={self.probs_.round(4)})"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_transitions_sklearn.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/transitions.py tests/test_transitions_sklearn.py
git commit -m "feat: add sklearn-style TransitionEstimator

TransitionEstimator.fit(df, state=, id=) returns self with:
- matrix_: transition probability matrix
- probs_: (theta_0, theta_1, theta_2)
- summary(): formatted output"
```

---

## Task 3: Create NFXP sklearn-Style Estimator

**Files:**
- Create: `src/econirl/estimators/nfxp.py`
- Create: `src/econirl/estimators/__init__.py`
- Test: `tests/test_nfxp_sklearn.py`

**Step 1: Write the failing test**

```python
# tests/test_nfxp_sklearn.py
import pytest
import pandas as pd
import numpy as np
import torch
from econirl.estimators import NFXP


def test_nfxp_init():
    """NFXP can be initialized with config."""
    est = NFXP(n_states=90, discount=0.9999)
    assert est.n_states == 90
    assert est.discount == 0.9999


def test_nfxp_fit_returns_self():
    """fit() returns self for chaining."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    result = est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert result is est


def test_nfxp_params_():
    """params_ is dict after fit."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert hasattr(est, "params_")
    assert isinstance(est.params_, dict)
    assert "theta_c" in est.params_
    assert "RC" in est.params_


def test_nfxp_se_():
    """se_ is dict after fit."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert hasattr(est, "se_")
    assert isinstance(est.se_, dict)


def test_nfxp_coef_():
    """coef_ is array (sklearn convention)."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert hasattr(est, "coef_")
    assert isinstance(est.coef_, np.ndarray)
    assert len(est.coef_) == 2


def test_nfxp_summary():
    """summary() returns formatted string."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    summary = est.summary()
    assert isinstance(summary, str)
    assert "theta_c" in summary
    assert "RC" in summary


def test_nfxp_transitions_():
    """transitions_ is available after fit."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert hasattr(est, "transitions_")
    assert est.transitions_.shape == (90, 90)


def test_nfxp_with_explicit_transitions():
    """Can provide pre-estimated transitions."""
    df = _make_simple_data()
    transitions = torch.eye(90)  # Dummy transitions

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id",
            transitions=transitions)

    assert torch.allclose(est.transitions_, transitions)


def _make_simple_data() -> pd.DataFrame:
    """Create simple test data."""
    np.random.seed(42)
    n_buses = 10
    n_periods = 50

    records = []
    for bus_id in range(1, n_buses + 1):
        mileage_bin = 0
        for period in range(1, n_periods + 1):
            # Simple replacement rule: replace when mileage > 50
            replaced = 1 if mileage_bin > 50 else 0

            records.append({
                "bus_id": bus_id,
                "period": period,
                "mileage_bin": mileage_bin,
                "replaced": replaced,
            })

            if replaced:
                mileage_bin = 0
            else:
                mileage_bin = min(mileage_bin + np.random.choice([0, 1, 2]), 89)

    return pd.DataFrame(records)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_nfxp_sklearn.py -v`
Expected: FAIL with "No module named 'econirl.estimators'"

**Step 3: Write minimal implementation**

```python
# src/econirl/estimators/__init__.py
"""sklearn-style estimators for DDC models."""

from econirl.estimators.nfxp import NFXP

__all__ = ["NFXP"]
```

```python
# src/econirl/estimators/nfxp.py
"""sklearn-style NFXP estimator for DDC models.

Example:
    >>> from econirl.estimators import NFXP
    >>> est = NFXP(n_states=90, discount=0.9999)
    >>> est.fit(df, state="mileage_bin", action="replaced", id="bus_id")
    >>> print(est.params_)
    >>> print(est.summary())
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch

from econirl.utilities import Utility, LinearCost


class NFXP:
    """NFXP estimator for dynamic discrete choice models.

    Nested Fixed Point algorithm from Rust (1987, 1988).

    Example:
        >>> est = NFXP(n_states=90, discount=0.9999)
        >>> est.fit(df, state="mileage_bin", action="replaced", id="bus_id")
        >>> print(est.params_)  # {'theta_c': 0.00107, 'RC': 9.35}
        >>> print(est.summary())

    Attributes (after fit):
        params_: dict of parameter estimates
        se_: dict of standard errors
        coef_: numpy array of coefficients (sklearn convention)
        log_likelihood_: maximized log-likelihood
        value_function_: value function V(s)
        transitions_: transition matrix
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | Utility | Callable = "linear_cost",
        se_method: Literal["asymptotic", "robust", "bootstrap"] = "robust",
        verbose: bool = False,
    ):
        """Initialize NFXP estimator.

        Args:
            n_states: Number of discrete states
            n_actions: Number of actions (default 2: keep/replace)
            discount: Discount factor beta
            utility: Utility specification. Can be:
                - "linear_cost": LinearCost utility (default)
                - Utility object: custom utility class
                - Callable: function (state, action, params) -> utility
            se_method: Standard error method
            verbose: Print progress
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.se_method = se_method
        self.verbose = verbose

        # Parse utility
        if utility == "linear_cost":
            self._utility = LinearCost()
        elif isinstance(utility, Utility):
            self._utility = utility
        elif callable(utility):
            from econirl.utilities import make_utility
            self._utility = make_utility(utility, n_params=2)
        else:
            raise ValueError(f"Unknown utility: {utility}")

        # Fitted attributes (None until fit)
        self.params_: dict | None = None
        self.se_: dict | None = None
        self.coef_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.value_function_: np.ndarray | None = None
        self.transitions_: torch.Tensor | None = None
        self.converged_: bool | None = None
        self._result = None  # Internal: full EstimationSummary

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        transitions: torch.Tensor | np.ndarray | None = None,
    ) -> "NFXP":
        """Fit the NFXP model to data.

        Args:
            data: DataFrame with panel data
            state: Column name for state variable
            action: Column name for action/choice
            id: Column name for individual identifier
            transitions: Pre-estimated transition matrix. If None, estimated from data.

        Returns:
            self
        """
        from econirl.core.types import DDCProblem, Panel, Trajectory
        from econirl.estimation.nfxp import NFXPEstimator
        from econirl.preferences.linear import LinearUtility
        from econirl.transitions import TransitionEstimator

        # Estimate transitions if not provided
        if transitions is None:
            trans_est = TransitionEstimator(n_states=self.n_states)
            trans_est.fit(data, state=state, id=id, action=action)
            self.transitions_ = trans_est.matrix_
        else:
            if isinstance(transitions, np.ndarray):
                transitions = torch.from_numpy(transitions).float()
            self.transitions_ = transitions

        # Build transition tensor for all actions
        # Action 0 (keep): use estimated transitions
        # Action 1 (replace): go to state 0 with prob 1
        P = torch.zeros(self.n_actions, self.n_states, self.n_states)
        P[0] = self.transitions_
        P[1, :, 0] = 1.0  # Replace -> state 0

        # Convert data to Panel
        panel = self._df_to_panel(data, state, action, id)

        # Create problem spec
        problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
        )

        # Create utility function (bridge to old API)
        utility = self._create_utility_bridge()

        # Run NFXP
        estimator = NFXPEstimator(
            se_method=self.se_method,
            verbose=self.verbose,
        )
        result = estimator.estimate(panel, utility, problem, P)

        # Store results
        self._result = result
        self.coef_ = result.parameters.numpy()
        self.params_ = dict(zip(result.parameter_names, self.coef_))
        self.se_ = dict(zip(result.parameter_names, result.standard_errors.numpy()))
        self.log_likelihood_ = result.log_likelihood
        self.value_function_ = result.value_function.numpy() if result.value_function is not None else None
        self.converged_ = result.converged

        return self

    def _df_to_panel(self, data: pd.DataFrame, state: str, action: str, id: str) -> "Panel":
        """Convert DataFrame to Panel object."""
        from econirl.core.types import Panel, Trajectory

        trajectories = []
        for ind_id in data[id].unique():
            ind_data = data[data[id] == ind_id].sort_index()
            states = torch.tensor(ind_data[state].values, dtype=torch.long)
            actions = torch.tensor(ind_data[action].values, dtype=torch.long)

            # Compute next states (shift by 1)
            if len(states) > 1:
                next_states = torch.cat([states[1:], states[-1:]])
            else:
                next_states = states.clone()

            traj = Trajectory(
                states=states,
                actions=actions,
                next_states=next_states,
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _create_utility_bridge(self):
        """Create bridge to old UtilityFunction API."""
        from econirl.preferences.linear import LinearUtility

        # Create feature matrix for LinearUtility
        # This matches LinearCost: u = -theta_c * s * (1-a) - RC * a
        feature_matrix = torch.zeros(self.n_states, self.n_actions, 2)
        for s in range(self.n_states):
            feature_matrix[s, 0, 0] = -s  # Keep: -theta_c * s
            feature_matrix[s, 0, 1] = 0   # Keep: no RC
            feature_matrix[s, 1, 0] = 0   # Replace: no theta_c
            feature_matrix[s, 1, 1] = -1  # Replace: -RC

        return LinearUtility(
            feature_matrix=feature_matrix,
            parameter_names=self._utility.param_names,
        )

    def summary(self) -> str:
        """Generate summary of estimation results."""
        if self._result is None:
            return "NFXP not fitted. Call fit() first."
        return self._result.summary()

    def to_latex(self, path: str | None = None) -> str:
        """Export results as LaTeX table."""
        if self._result is None:
            raise ValueError("NFXP not fitted. Call fit() first.")
        return self._result.to_latex(path)

    def predict_proba(self, states: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict choice probabilities for given states.

        Args:
            states: Array of state indices

        Returns:
            Array of shape (len(states), n_actions) with choice probabilities
        """
        if self._result is None:
            raise ValueError("NFXP not fitted. Call fit() first.")

        policy = self._result.policy
        if isinstance(states, torch.Tensor):
            states = states.numpy()

        return policy[states].numpy()

    def __repr__(self) -> str:
        fitted = self.params_ is not None
        return f"NFXP(n_states={self.n_states}, discount={self.discount}, fitted={fitted})"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_nfxp_sklearn.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/estimators/ tests/test_nfxp_sklearn.py
git commit -m "feat: add sklearn-style NFXP estimator

NFXP(config).fit(df, state=, action=, id=) returns self with:
- params_: dict of estimates
- se_: dict of standard errors
- coef_: numpy array (sklearn convention)
- summary(): formatted output
- predict_proba(): choice probabilities"
```

---

## Task 4: Add simulate() and counterfactual() Methods

**Files:**
- Modify: `src/econirl/estimators/nfxp.py:150-200`
- Test: `tests/test_nfxp_sklearn.py` (add tests)

**Step 1: Write the failing test**

Add to `tests/test_nfxp_sklearn.py`:

```python
def test_nfxp_simulate():
    """simulate() returns DataFrame with simulated data."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    sim = est.simulate(n_agents=10, n_periods=20, seed=42)

    assert isinstance(sim, pd.DataFrame)
    assert len(sim) == 10 * 20
    assert "agent_id" in sim.columns
    assert "period" in sim.columns
    assert "state" in sim.columns
    assert "action" in sim.columns


def test_nfxp_counterfactual():
    """counterfactual() computes outcomes under different parameters."""
    df = _make_simple_data()

    est = NFXP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    # Double the replacement cost
    cf = est.counterfactual(RC=est.params_["RC"] * 2)

    assert hasattr(cf, "value_function")
    assert hasattr(cf, "policy")
    assert cf.value_function.shape == (90,)
    assert cf.policy.shape == (90, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_nfxp_sklearn.py::test_nfxp_simulate -v`
Expected: FAIL with "AttributeError: 'NFXP' object has no attribute 'simulate'"

**Step 3: Add implementation to NFXP class**

Add these methods to `src/econirl/estimators/nfxp.py`:

```python
    def simulate(
        self,
        n_agents: int,
        n_periods: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Simulate choices under estimated policy.

        Args:
            n_agents: Number of agents to simulate
            n_periods: Number of periods per agent
            seed: Random seed for reproducibility

        Returns:
            DataFrame with columns: agent_id, period, state, action
        """
        if self._result is None:
            raise ValueError("NFXP not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        policy = self._result.policy  # Shape: (n_states, n_actions)
        P = self.transitions_  # Shape: (n_states, n_states)

        records = []
        for agent_id in range(1, n_agents + 1):
            state = 0
            for period in range(1, n_periods + 1):
                # Sample action from policy
                probs = policy[state].numpy()
                action = np.random.choice(self.n_actions, p=probs)

                records.append({
                    "agent_id": agent_id,
                    "period": period,
                    "state": state,
                    "action": action,
                })

                # Transition
                if action == 1:  # Replace
                    state = 0
                else:
                    # Sample next state from transition matrix
                    trans_probs = P[state].numpy()
                    state = np.random.choice(self.n_states, p=trans_probs)

        return pd.DataFrame(records)

    def counterfactual(self, **param_changes) -> "CounterfactualResult":
        """Compute counterfactual outcomes under different parameters.

        Args:
            **param_changes: Parameter values to change, e.g., RC=15.0

        Returns:
            CounterfactualResult with value_function, policy
        """
        if self._result is None:
            raise ValueError("NFXP not fitted. Call fit() first.")

        from econirl.core.bellman import SoftBellmanOperator
        from econirl.core.solvers import policy_iteration
        from econirl.core.types import DDCProblem

        # Build new parameters
        new_params = self.params_.copy()
        new_params.update(param_changes)
        params_tensor = torch.tensor([new_params[name] for name in self._utility.param_names])

        # Compute new utility matrix
        U = self._utility.matrix(self.n_states, params_tensor)

        # Build full transition tensor
        P = torch.zeros(self.n_actions, self.n_states, self.n_states)
        P[0] = self.transitions_
        P[1, :, 0] = 1.0

        # Create problem and solve
        problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
        )
        operator = SoftBellmanOperator(problem, P)
        result = policy_iteration(operator, U)

        return CounterfactualResult(
            params=new_params,
            value_function=result.V.numpy(),
            policy=result.policy.numpy(),
        )


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    params: dict
    value_function: np.ndarray
    policy: np.ndarray

    def __repr__(self) -> str:
        return f"CounterfactualResult(params={self.params})"
```

Add import at top of file:
```python
from dataclasses import dataclass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_nfxp_sklearn.py::test_nfxp_simulate tests/test_nfxp_sklearn.py::test_nfxp_counterfactual -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/estimators/nfxp.py tests/test_nfxp_sklearn.py
git commit -m "feat: add simulate() and counterfactual() to NFXP

- simulate(n_agents, n_periods) returns DataFrame with simulated data
- counterfactual(**params) returns CounterfactualResult with new V and policy"
```

---

## Task 5: Add CCP sklearn-Style Estimator

**Files:**
- Create: `src/econirl/estimators/ccp.py`
- Modify: `src/econirl/estimators/__init__.py`
- Test: `tests/test_ccp_sklearn.py`

**Step 1: Write the failing test**

```python
# tests/test_ccp_sklearn.py
import pytest
import pandas as pd
import numpy as np
from econirl.estimators import CCP


def test_ccp_init():
    """CCP can be initialized with config."""
    est = CCP(n_states=90, discount=0.9999)
    assert est.n_states == 90
    assert est.discount == 0.9999


def test_ccp_fit_returns_self():
    """fit() returns self for chaining."""
    df = _make_simple_data()

    est = CCP(n_states=90, discount=0.9999)
    result = est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert result is est


def test_ccp_params_():
    """params_ is dict after fit."""
    df = _make_simple_data()

    est = CCP(n_states=90, discount=0.9999)
    est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    assert hasattr(est, "params_")
    assert isinstance(est.params_, dict)
    assert "theta_c" in est.params_
    assert "RC" in est.params_


def test_ccp_same_interface_as_nfxp():
    """CCP has same interface as NFXP."""
    from econirl.estimators import NFXP

    df = _make_simple_data()

    nfxp = NFXP(n_states=90, discount=0.9999)
    nfxp.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    ccp = CCP(n_states=90, discount=0.9999)
    ccp.fit(df, state="mileage_bin", action="replaced", id="bus_id")

    # Same attributes
    assert set(nfxp.params_.keys()) == set(ccp.params_.keys())
    assert hasattr(ccp, "se_")
    assert hasattr(ccp, "coef_")
    assert hasattr(ccp, "summary")


def _make_simple_data() -> pd.DataFrame:
    """Create simple test data."""
    np.random.seed(42)
    n_buses = 10
    n_periods = 50

    records = []
    for bus_id in range(1, n_buses + 1):
        mileage_bin = 0
        for period in range(1, n_periods + 1):
            replaced = 1 if mileage_bin > 50 else 0

            records.append({
                "bus_id": bus_id,
                "period": period,
                "mileage_bin": mileage_bin,
                "replaced": replaced,
            })

            if replaced:
                mileage_bin = 0
            else:
                mileage_bin = min(mileage_bin + np.random.choice([0, 1, 2]), 89)

    return pd.DataFrame(records)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ccp_sklearn.py -v`
Expected: FAIL with "cannot import name 'CCP'"

**Step 3: Write minimal implementation**

```python
# src/econirl/estimators/ccp.py
"""sklearn-style CCP estimator for DDC models.

Hotz-Miller CCP estimator that inverts the conditional choice probabilities.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch

from econirl.utilities import Utility, LinearCost
from econirl.estimators.nfxp import NFXP, CounterfactualResult


class CCP(NFXP):
    """CCP (Hotz-Miller) estimator for dynamic discrete choice models.

    Uses conditional choice probability inversion rather than solving
    the full dynamic programming problem.

    Same interface as NFXP - just uses different estimation method internally.

    Example:
        >>> est = CCP(n_states=90, discount=0.9999)
        >>> est.fit(df, state="mileage_bin", action="replaced", id="bus_id")
        >>> print(est.params_)
    """

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        transitions: torch.Tensor | np.ndarray | None = None,
    ) -> "CCP":
        """Fit the CCP model to data.

        Same interface as NFXP.fit().
        """
        from econirl.core.types import DDCProblem, Panel
        from econirl.estimation.ccp import CCPEstimator
        from econirl.transitions import TransitionEstimator

        # Estimate transitions if not provided
        if transitions is None:
            trans_est = TransitionEstimator(n_states=self.n_states)
            trans_est.fit(data, state=state, id=id, action=action)
            self.transitions_ = trans_est.matrix_
        else:
            if isinstance(transitions, np.ndarray):
                transitions = torch.from_numpy(transitions).float()
            self.transitions_ = transitions

        # Build transition tensor
        P = torch.zeros(self.n_actions, self.n_states, self.n_states)
        P[0] = self.transitions_
        P[1, :, 0] = 1.0

        # Convert data to Panel
        panel = self._df_to_panel(data, state, action, id)

        # Create problem spec
        problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
        )

        # Create utility function
        utility = self._create_utility_bridge()

        # Run CCP
        estimator = CCPEstimator(
            se_method=self.se_method,
            verbose=self.verbose,
        )
        result = estimator.estimate(panel, utility, problem, P)

        # Store results
        self._result = result
        self.coef_ = result.parameters.numpy()
        self.params_ = dict(zip(result.parameter_names, self.coef_))
        self.se_ = dict(zip(result.parameter_names, result.standard_errors.numpy()))
        self.log_likelihood_ = result.log_likelihood
        self.value_function_ = result.value_function.numpy() if result.value_function is not None else None
        self.converged_ = result.converged

        return self

    def __repr__(self) -> str:
        fitted = self.params_ is not None
        return f"CCP(n_states={self.n_states}, discount={self.discount}, fitted={fitted})"
```

Update `src/econirl/estimators/__init__.py`:

```python
"""sklearn-style estimators for DDC models."""

from econirl.estimators.nfxp import NFXP, CounterfactualResult
from econirl.estimators.ccp import CCP

__all__ = ["NFXP", "CCP", "CounterfactualResult"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ccp_sklearn.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/estimators/ tests/test_ccp_sklearn.py
git commit -m "feat: add sklearn-style CCP estimator

CCP estimator with same interface as NFXP.
Uses Hotz-Miller CCP inversion internally."
```

---

## Task 6: Update Package Exports

**Files:**
- Modify: `src/econirl/__init__.py`

**Step 1: Write the failing test**

```python
# tests/test_package_imports.py
import pytest


def test_import_nfxp():
    """Can import NFXP from econirl."""
    from econirl import NFXP
    assert NFXP is not None


def test_import_ccp():
    """Can import CCP from econirl."""
    from econirl import CCP
    assert CCP is not None


def test_import_utilities():
    """Can import utility classes from econirl."""
    from econirl import LinearCost, Utility
    assert LinearCost is not None
    assert Utility is not None


def test_import_transition_estimator():
    """Can import TransitionEstimator from econirl."""
    from econirl import TransitionEstimator
    assert TransitionEstimator is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py -v`
Expected: FAIL with "cannot import name 'NFXP'"

**Step 3: Update __init__.py**

```python
# src/econirl/__init__.py
"""
econirl: The StatsModels of IRL

A Python package bridging Structural Econometrics and Inverse Reinforcement Learning.
Provides economist-friendly APIs for estimating dynamic discrete choice models with
rich statistical inference.

Key Features:
- scikit-learn style API: Estimator(config).fit(data)
- Multiple estimation methods (NFXP, CCP, NPL)
- Rich statistical inference (standard errors, hypothesis tests)
- Simulation and counterfactual analysis

Example:
    >>> from econirl import NFXP
    >>> from econirl.datasets import load_rust_bus
    >>>
    >>> df = load_rust_bus()
    >>> est = NFXP(n_states=90, discount=0.9999)
    >>> est.fit(df, state="mileage_bin", action="replaced", id="bus_id")
    >>> print(est.summary())
"""

__version__ = "0.1.0"

# sklearn-style estimators (new API)
from econirl.estimators import NFXP, CCP

# Utilities
from econirl.utilities import Utility, LinearCost, make_utility

# Transitions
from econirl.transitions import TransitionEstimator

# Core types
from econirl.core.types import DDCProblem, Panel, Trajectory

# Environments (for simulation)
from econirl.environments.rust_bus import RustBusEnvironment

# Legacy estimators (old API, for backward compatibility)
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator

# Preferences (old API)
from econirl.preferences.linear import LinearUtility

# Datasets
from econirl import datasets

# Replication
from econirl import replication

__all__ = [
    # Version
    "__version__",
    # sklearn-style estimators (new API)
    "NFXP",
    "CCP",
    # Utilities
    "Utility",
    "LinearCost",
    "make_utility",
    # Transitions
    "TransitionEstimator",
    # Core types
    "DDCProblem",
    "Panel",
    "Trajectory",
    # Environments
    "RustBusEnvironment",
    # Legacy (old API)
    "NFXPEstimator",
    "CCPEstimator",
    "LinearUtility",
    # Submodules
    "datasets",
    "replication",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/__init__.py tests/test_package_imports.py
git commit -m "feat: update package exports for sklearn-style API

Top-level imports now include:
- NFXP, CCP (new sklearn-style estimators)
- LinearCost, Utility, make_utility
- TransitionEstimator

Old API (NFXPEstimator, etc.) still available for backward compatibility."
```

---

## Task 7: Add Integration Test with Real Data

**Files:**
- Create: `tests/integration/test_sklearn_api.py`

**Step 1: Write the test**

```python
# tests/integration/test_sklearn_api.py
"""Integration tests for sklearn-style API with real data."""

import pytest
import numpy as np
import pandas as pd
from econirl import NFXP, CCP, TransitionEstimator
from econirl.datasets import load_rust_bus


@pytest.fixture
def rust_data():
    """Load Rust bus data."""
    return load_rust_bus(original=True)


class TestNFXPIntegration:
    """Integration tests for NFXP estimator."""

    def test_full_workflow(self, rust_data):
        """Test complete NFXP workflow on real data."""
        df = rust_data

        # Fit
        est = NFXP(n_states=90, discount=0.9999)
        est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        # Check estimates are reasonable
        assert est.params_["theta_c"] > 0
        assert est.params_["RC"] > 0
        assert est.converged_

        # Check standard errors
        assert all(se > 0 for se in est.se_.values())

        # Summary works
        summary = est.summary()
        assert "theta_c" in summary
        assert "RC" in summary

    def test_simulate(self, rust_data):
        """Test simulation from fitted model."""
        df = rust_data

        est = NFXP(n_states=90, discount=0.9999)
        est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        sim = est.simulate(n_agents=100, n_periods=50, seed=42)

        # Check simulation output
        assert len(sim) == 100 * 50
        assert sim["action"].mean() > 0  # Some replacements occur
        assert sim["state"].max() < 90   # States in valid range

    def test_counterfactual(self, rust_data):
        """Test counterfactual analysis."""
        df = rust_data

        est = NFXP(n_states=90, discount=0.9999)
        est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        # Double RC
        cf = est.counterfactual(RC=est.params_["RC"] * 2)

        # Higher RC should mean less replacement
        # (lower probability of action 1 at each state)
        baseline_replace_prob = est._result.policy[:, 1].numpy()
        cf_replace_prob = cf.policy[:, 1]

        # On average, replacement should be less likely with higher RC
        assert cf_replace_prob.mean() < baseline_replace_prob.mean()


class TestCCPIntegration:
    """Integration tests for CCP estimator."""

    def test_full_workflow(self, rust_data):
        """Test complete CCP workflow on real data."""
        df = rust_data

        est = CCP(n_states=90, discount=0.9999)
        est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        # Check estimates are reasonable
        assert est.params_["theta_c"] > 0
        assert est.params_["RC"] > 0

    def test_compare_to_nfxp(self, rust_data):
        """CCP and NFXP should give similar estimates."""
        df = rust_data

        nfxp = NFXP(n_states=90, discount=0.9999)
        nfxp.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        ccp = CCP(n_states=90, discount=0.9999)
        ccp.fit(df, state="mileage_bin", action="replaced", id="bus_id")

        # Estimates should be in same ballpark (within 50%)
        for param in ["theta_c", "RC"]:
            ratio = ccp.params_[param] / nfxp.params_[param]
            assert 0.5 < ratio < 2.0, f"{param}: CCP={ccp.params_[param]}, NFXP={nfxp.params_[param]}"


class TestTransitionEstimatorIntegration:
    """Integration tests for TransitionEstimator."""

    def test_rust_data_transitions(self, rust_data):
        """Test transition estimation on real data."""
        df = rust_data

        est = TransitionEstimator(n_states=90)
        est.fit(df, state="mileage_bin", id="bus_id", action="replaced")

        # Probabilities should sum to 1
        assert sum(est.probs_) == pytest.approx(1.0)

        # Rust data has specific transition pattern
        # theta_0 ≈ 0.39, theta_1 ≈ 0.60, theta_2 ≈ 0.01
        assert est.probs_[0] > 0.3  # Some stay
        assert est.probs_[1] > 0.5  # Most move +1
        assert est.probs_[2] < 0.1  # Few move +2
```

**Step 2: Run test**

Run: `pytest tests/integration/test_sklearn_api.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_sklearn_api.py
git commit -m "test: add integration tests for sklearn-style API

Tests NFXP, CCP, TransitionEstimator on real Rust bus data.
Verifies full workflow, simulation, counterfactual analysis."
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/tutorials/quickstart.rst` (if docs exist)

**Step 1: Update README.md**

Add new Quick Start section:

```markdown
## Quick Start

```python
from econirl import NFXP
from econirl.datasets import load_rust_bus

# Load data
df = load_rust_bus()

# Fit model
est = NFXP(n_states=90, discount=0.9999)
est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

# View results
print(est.params_)   # {'theta_c': 0.00107, 'RC': 9.35}
print(est.summary())

# Simulate
sim = est.simulate(n_agents=100, n_periods=50)

# Counterfactual: what if RC doubled?
cf = est.counterfactual(RC=est.params_["RC"] * 2)
```

### Available Estimators

| Estimator | Description |
|-----------|-------------|
| `NFXP` | Nested Fixed Point (Rust 1987) |
| `CCP` | Conditional Choice Probability (Hotz-Miller) |

All estimators share the same interface:
- `est.fit(df, state=, action=, id=)` - fit model
- `est.params_` - parameter estimates
- `est.se_` - standard errors
- `est.summary()` - formatted results
- `est.simulate()` - simulate choices
- `est.counterfactual()` - policy analysis
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with sklearn-style API quickstart"
```

---

## Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Fix any failures**

If any tests fail, fix them before proceeding.

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: ensure all tests pass with new sklearn API"
```

---

## Summary

After completing all tasks, the package will have:

1. **New sklearn-style API:**
   - `NFXP(config).fit(df, state=, action=, id=)`
   - `CCP(config).fit(df, state=, action=, id=)`
   - `TransitionEstimator().fit(df, state=, id=)`

2. **Utility classes:**
   - `LinearCost` (built-in)
   - `Utility` base class
   - `make_utility()` factory

3. **Results as properties:**
   - `params_`, `se_`, `coef_` (sklearn convention)
   - `summary()`, `to_latex()`
   - `simulate()`, `counterfactual()`

4. **Backward compatibility:**
   - Old `NFXPEstimator` still available
   - Old `LinearUtility` still available
