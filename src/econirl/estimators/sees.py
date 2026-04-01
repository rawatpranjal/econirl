"""Sklearn-style SEES estimator for dynamic discrete choice models.

This module provides a SEES class with a scikit-learn style API that wraps
the underlying SEESEstimator from econirl.estimation.sees. It accepts pandas
DataFrames with column names instead of the low-level Panel API.

SEES (Sieve Estimation of Economic Structural models, Luo & Sang 2024)
approximates V(s) with sieve basis functions (Fourier or polynomial),
then jointly optimizes structural parameters theta and basis coefficients
alpha via penalized MLE. This avoids the costly inner fixed-point loop
of NFXP and the neural network training of NNES.

Example:
    >>> from econirl.estimators import SEES
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit
    >>> model = SEES(n_states=90, discount=0.9999, basis_type="fourier", basis_dim=8)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"theta_c": 0.001, "RC": 3.01}
    >>> print(model.summary())
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.sees import SEESConfig, SEESEstimator
from econirl.preferences.linear import LinearUtility
from econirl.transitions import TransitionEstimator


class SEES:
    """Sklearn-style SEES estimator for dynamic discrete choice models.

    SEES (Sieve Estimation of Economic Structural models) approximates the
    value function V(s) with sieve basis functions and jointly optimizes
    structural parameters and basis coefficients via penalized MLE. This
    avoids the inner fixed-point loop of NFXP while using a closed-form
    basis expansion instead of neural network training (Luo & Sang 2024).

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
    se_method : str, default="asymptotic"
        Method for computing standard errors.
    basis_type : str, default="fourier"
        Sieve basis type. Options: "fourier", "polynomial".
    basis_dim : int, default=8
        Number of basis functions for the value function approximation.
    penalty_lambda : float, default=0.01
        L2 penalty on basis coefficients alpha.
    max_iter : int, default=500
        Maximum L-BFGS-B iterations.
    verbose : bool, default=False
        Whether to print progress messages during estimation.

    Attributes
    ----------
    params_ : dict
        Estimated parameters after fitting.
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
    alpha_ : numpy.ndarray or None
        Estimated basis coefficients after fitting, shape (basis_dim,).
    converged_ : bool
        Whether the optimization converged.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    References
    ----------
    Luo, Y. and Sang, Y. (2024). "Sieve Estimation of Dynamic Discrete
        Choice Models." Working Paper.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | RewardSpec = "linear_cost",
        se_method: Literal["robust", "asymptotic"] = "asymptotic",
        basis_type: str = "fourier",
        basis_dim: int = 8,
        penalty_lambda: float = 0.01,
        max_iter: int = 500,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.utility = utility
        self.se_method = se_method
        self.basis_type = basis_type
        self.basis_dim = basis_dim
        self.penalty_lambda = penalty_lambda
        self.max_iter = max_iter
        self.verbose = verbose

        # Fitted attributes (set after fit())
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.value_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.reward_spec_: RewardSpec | None = None
        self.alpha_: np.ndarray | None = None

        # Internal storage
        self._result = None
        self._panel = None
        self._utility_fn = None
        self._problem = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        reward: RewardSpec | None = None,
    ) -> "SEES":
        """Fit the SEES estimator to data.

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
        self : SEES
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

        # Build full transition matrices as JAX array (SEES uses JAX)
        transition_tensor = self._build_transition_tensor(self.transitions_)

        # Create problem specification
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        # Create the underlying SEES estimator
        config = SEESConfig(
            basis_type=self.basis_type,
            basis_dim=self.basis_dim,
            penalty_lambda=self.penalty_lambda,
            max_iter=self.max_iter,
            compute_se=True,
            se_method=self.se_method,
            verbose=self.verbose,
        )
        estimator = SEESEstimator(config=config)

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

    def _build_transition_tensor(self, keep_transitions: np.ndarray) -> jnp.ndarray:
        """Build full transition tensor for both actions.

        Parameters
        ----------
        keep_transitions : numpy.ndarray
            Transition matrix for action=0 (keep), shape (n_states, n_states).

        Returns
        -------
        jnp.ndarray
            Transition tensor of shape (n_actions, n_states, n_states).
        """
        n = self.n_states
        transitions = np.zeros((self.n_actions, n, n), dtype=np.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = keep_transitions.astype(np.float32)

        # Action 1 (replace): reset to state 0, then transition
        for s in range(n):
            transitions[1, s, :] = transitions[0, 0, :]

        return jnp.array(transitions)

    def _create_utility(self) -> LinearUtility:
        """Create utility function for estimation.

        Returns
        -------
        LinearUtility
            Utility function with appropriate features.
        """
        if self.utility != "linear_cost":
            raise ValueError(f"Unknown utility specification: {self.utility}")

        n = self.n_states
        features = jnp.zeros((n, self.n_actions, 2))

        mileage = jnp.arange(n, dtype=jnp.float32)

        # Keep action (a=0): feature = [-s, 0]
        features = features.at[:, 0, 0].set(-mileage)
        # Replace action (a=1): feature = [0, -1]
        features = features.at[:, 1, 1].set(-1.0)

        return LinearUtility(
            feature_matrix=features,
            parameter_names=["theta_c", "RC"],
        )

    def _extract_results(self) -> None:
        """Extract results from estimation into sklearn-style attributes."""
        if self._result is None:
            return

        # Parameter estimates
        params = np.asarray(self._result.parameters)
        param_names = self._result.parameter_names

        self.params_ = {name: float(val) for name, val in zip(param_names, params)}
        self.coef_ = params.copy()

        # Standard errors
        se = np.asarray(self._result.standard_errors)
        self.se_ = {name: float(val) for name, val in zip(param_names, se)}

        # Other attributes
        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)

        if self._result.value_function is not None:
            self.value_ = np.asarray(self._result.value_function)

        if self._result.policy is not None:
            self.policy_ = np.asarray(self._result.policy)

        # p-values from t-statistics
        if self.se_ is not None:
            pvalues: dict[str, float] = {}
            for name in self.params_:
                se_val = self.se_[name]
                if se_val and se_val > 0:
                    t_stat = self.params_[name] / se_val
                    pvalues[name] = float(
                        2 * (1 - scipy_norm.cdf(abs(t_stat)))
                    )
                else:
                    pvalues[name] = float("nan")
            self.pvalues_ = pvalues

        # SEES-specific: basis coefficients
        if self._result.metadata:
            alpha = self._result.metadata.get("alpha")
            if alpha is not None:
                self.alpha_ = np.asarray(alpha)

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "SEES: Not fitted yet. Call fit() first."

        return self._result.summary()

    def conf_int(self, alpha: float = 0.05) -> dict:
        """Compute confidence intervals for parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.  Returns (1 - alpha) confidence intervals.

        Returns
        -------
        dict
            ``{param_name: (lower, upper)}`` confidence intervals.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.params_ is None or self.se_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        z = scipy_norm.ppf(1 - alpha / 2)
        intervals: dict[str, tuple[float, float]] = {}
        for name in self.params_:
            est = self.params_[name]
            se = self.se_[name]
            intervals[name] = (est - z * se, est + z * se)
        return intervals

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Predict choice probabilities for given states.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.

        Returns
        -------
        numpy.ndarray
            Choice probabilities of shape (len(states), n_actions).
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)
        policy = np.asarray(self._result.policy)
        return policy[states]

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"SEES(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, basis_type='{self.basis_type}', "
                f"basis_dim={self.basis_dim}, fitted=True)"
            )
        return (
            f"SEES(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, basis_type='{self.basis_type}', "
            f"basis_dim={self.basis_dim}, fitted=False)"
        )
