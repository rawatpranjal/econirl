"""Sklearn-style CCP estimator for dynamic discrete choice models.

This module provides a CCP class with a scikit-learn style API that wraps
the underlying CCPEstimator from econirl.estimation.ccp. It implements the
Hotz-Miller (1993) approach and NPL extension (Aguirregabiria-Mira 2002).

The CCP estimator has the same interface as NFXP, allowing users to easily
switch between estimation methods.

Example:
    >>> from econirl.estimators import CCP
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit (Hotz-Miller by default)
    >>> model = CCP(n_states=90, discount=0.9999)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"theta_c": 0.001, "RC": 9.35}
    >>> print(model.summary())
    >>>
    >>> # Use NPL (iterative refinement)
    >>> model_npl = CCP(n_states=90, num_policy_iterations=10)
    >>> model_npl.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.ccp import CCPEstimator
from econirl.estimators.nfxp import NFXP
from econirl.transitions import TransitionEstimator


class CCP(NFXP):
    """Sklearn-style CCP estimator for dynamic discrete choice models.

    The CCP (Conditional Choice Probability) approach estimates utility
    parameters using the Hotz-Miller inversion theorem. This is typically
    faster than NFXP because it avoids the nested fixed-point computation.

    With num_policy_iterations=1, this is the classic Hotz-Miller estimator.
    With num_policy_iterations>1, this becomes the NPL (Nested Pseudo Likelihood)
    algorithm that iterates to the MLE.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states (e.g., mileage bins).
    n_actions : int, default=2
        Number of discrete actions (e.g., keep/replace).
    discount : float, default=0.9999
        Time discount factor (beta).
    utility : str or RewardSpec, default="linear_cost"
        Utility specification.  Pass ``"linear_cost"`` for the classic Rust
        bus model (``u = -theta_c * s * (1-a) - RC * a``), or a
        ``RewardSpec`` for custom features.
    se_method : str, default="robust"
        Method for computing standard errors. Options: "robust", "asymptotic".
    verbose : bool, default=False
        Whether to print progress messages during estimation.
    num_policy_iterations : int, default=1
        Number of policy iterations. K=1 is Hotz-Miller, K>1 is NPL.
        Set to -1 for convergence-based stopping.

    Attributes
    ----------
    params_ : dict
        Estimated parameters after fitting. Keys are parameter names
        (e.g., "theta_c", "RC") and values are point estimates.
    se_ : dict
        Standard errors for each parameter.
    coef_ : numpy.ndarray
        Coefficients as a numpy array (sklearn convention).
    log_likelihood_ : float
        Maximized log-likelihood value.
    pvalues_ : dict
        P-values for each parameter (from Wald t-test).
    policy_ : numpy.ndarray
        Estimated choice probabilities P(a|s) of shape (n_states, n_actions).
    value_ : numpy.ndarray
        Estimated value function V(s) of shape (n_states,).
    value_function_ : numpy.ndarray
        Alias for ``value_`` (backward compatibility).
    transitions_ : numpy.ndarray
        Transition probability matrix (n_states x n_states).
    converged_ : bool
        Whether the optimization converged.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    Examples
    --------
    >>> from econirl.estimators import CCP
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "bus_id": [0, 0, 1, 1],
    ...     "mileage": [10, 20, 15, 30],
    ...     "replaced": [0, 0, 0, 1],
    ... })
    >>>
    >>> # Hotz-Miller (fast, one-step)
    >>> model = CCP(n_states=90)
    >>> model.fit(df, state="mileage", action="replaced", id="bus_id")
    >>>
    >>> # NPL (iterates towards MLE)
    >>> model_npl = CCP(n_states=90, num_policy_iterations=10)
    >>> model_npl.fit(df, state="mileage", action="replaced", id="bus_id")

    References
    ----------
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Aguirregabiria, V. and Mira, P. (2002). "Swapping the Nested Fixed Point
        Algorithm." Econometrica 70(4), 1519-1543.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | RewardSpec = "linear_cost",
        se_method: Literal["robust", "asymptotic"] = "robust",
        verbose: bool = False,
        num_policy_iterations: int = 1,
    ):
        """Initialize the CCP estimator.

        Parameters
        ----------
        n_states : int, default=90
            Number of discrete states.
        n_actions : int, default=2
            Number of discrete actions.
        discount : float, default=0.9999
            Time discount factor (beta).
        utility : str or RewardSpec, default="linear_cost"
            Utility specification to use.  Pass ``"linear_cost"`` for the
            classic Rust bus model, or a ``RewardSpec`` for custom features.
        se_method : str, default="robust"
            Method for computing standard errors.
        verbose : bool, default=False
            Whether to print progress messages.
        num_policy_iterations : int, default=1
            Number of NPL iterations (K=1 is Hotz-Miller).
        """
        # Initialize parent with shared parameters
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            utility=utility,
            se_method=se_method,
            verbose=verbose,
        )
        # CCP-specific parameter
        self.num_policy_iterations = num_policy_iterations

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        reward: RewardSpec | None = None,
    ) -> "CCP":
        """Fit the CCP estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with observations.  When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
            When a Panel/TrajectoryPanel is passed, column names are ignored.
        state : str, optional
            Column name for the state variable (required for DataFrame input).
        action : str, optional
            Column name for the action variable (required for DataFrame input).
        id : str, optional
            Column name for the individual identifier (required for DataFrame
            input).
        transitions : numpy.ndarray, optional
            Pre-estimated transition matrix of shape (n_states, n_states).
            If None, transitions are estimated from the data.
        reward : RewardSpec, optional
            Reward/utility specification.  If provided, overrides the
            ``utility`` parameter passed at construction time.

        Returns
        -------
        self : CCP
            Returns self for method chaining.
        """
        # Resolve reward spec: explicit argument > constructor parameter
        reward_spec = reward if reward is not None else self.utility

        # --- Handle data: DataFrame or Panel/TrajectoryPanel ---
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            self._panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, "
                f"got {type(data)}"
            )

        # --- Handle reward: RewardSpec or string ---
        if isinstance(reward_spec, RewardSpec):
            self.reward_spec_ = reward_spec
            self._utility_fn = reward_spec.to_linear_utility()
        elif reward_spec == "linear_cost":
            self._utility_fn = self._create_utility()
            # Also create RewardSpec from the utility for consistency
            self.reward_spec_ = RewardSpec(
                self._utility_fn.feature_matrix,
                self._utility_fn.parameter_names,
            )
        else:
            raise ValueError(f"Unknown reward/utility specification: {reward_spec}")

        # Estimate transitions if not provided
        if transitions is None:
            trans_estimator = TransitionEstimator(
                n_states=self.n_states,
                max_increase=2,
            )
            trans_estimator.fit(self._panel)
            self.transitions_ = trans_estimator.matrix_
        else:
            self.transitions_ = np.asarray(transitions)

        # Build full transition matrices (for both actions)
        transition_tensor = self._build_transition_tensor(self.transitions_)

        # Create problem specification
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        # Create the underlying CCP estimator (this is the key difference from NFXP)
        estimator = CCPEstimator(
            num_policy_iterations=self.num_policy_iterations,
            se_method=self.se_method,
            verbose=self.verbose,
        )

        # Run estimation
        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._utility_fn,
            problem=self._problem,
            transitions=transition_tensor,
        )

        # Extract results
        self._extract_results()

        return self

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "CCP: Not fitted yet. Call fit() first."

        return self._result.summary()

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"CCP(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, num_policy_iterations={self.num_policy_iterations}, "
                f"fitted=True)"
            )
        return (
            f"CCP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, num_policy_iterations={self.num_policy_iterations}, "
            f"fitted=False)"
        )
