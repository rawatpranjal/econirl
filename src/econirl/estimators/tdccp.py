"""Sklearn-style TD-CCP estimator for dynamic discrete choice models.

This module provides a TDCCP class with a scikit-learn style API that wraps
the underlying TDCCPEstimator from econirl.estimation.td_ccp. It accepts pandas
DataFrames with column names instead of the low-level Panel API.

TD-CCP (Temporal Difference CCP) is the neural extension of CCP. Instead of
matrix inversion, it trains neural EV component networks via semi-gradient
TD learning. The approach combines CCP estimation with approximate value
iteration (AVI), decomposing the expected value function into per-feature
components each learned by a separate MLP.

The killer feature of TD-CCP is the ``ev_features_`` attribute after fitting,
which shows how much of the continuation value comes from each structural
feature -- providing interpretable decomposition of forward-looking behavior.

Example:
    >>> from econirl.estimators import TDCCP
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit
    >>> model = TDCCP(n_states=90, discount=0.9999, utility="linear_cost")
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"theta_c": 0.001, "RC": 9.35}
    >>> print(model.summary())
    >>>
    >>> # Interpretable EV decomposition
    >>> print(model.ev_features_)   # shape (n_states, n_features)
    >>>
    >>> # Predict choice probabilities
    >>> proba = model.predict_proba(states=np.array([0, 10, 50]))
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator
from econirl.preferences.linear import LinearUtility
from econirl.transitions import TransitionEstimator


class TDCCP:
    """Sklearn-style TD-CCP estimator for dynamic discrete choice models.

    TD-CCP (Temporal Difference CCP) estimates utility parameters using
    neural network-based approximate value iteration combined with CCP
    decomposition. Instead of matrix inversion (as in standard CCP), it
    trains per-feature EV component networks via semi-gradient TD learning,
    then uses the learned components in a partial MLE for structural
    parameters.

    This is particularly useful for large state spaces where matrix-based
    CCP methods are computationally infeasible.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states (e.g., mileage bins).
    n_actions : int, default=2
        Number of discrete actions (e.g., keep/replace).
    discount : float, default=0.9999
        Time discount factor (beta).
    utility : str or RewardSpec, default="linear_cost"
        Utility specification. Pass ``"linear_cost"`` for the classic Rust
        bus model (``u = -theta_c * s * (1-a) - RC * a``), or a
        ``RewardSpec`` for custom features.
    se_method : str, default="asymptotic"
        Method for computing standard errors. Options: "robust", "asymptotic".
    hidden_dim : int, default=64
        Number of hidden units per layer in the EV component networks.
    num_hidden_layers : int, default=2
        Number of hidden layers in the EV component networks.
    avi_iterations : int, default=20
        Number of approximate value iteration rounds.
    epochs_per_avi : int, default=30
        Number of SGD epochs per AVI iteration.
    learning_rate : float, default=1e-3
        Learning rate for neural network training.
    batch_size : int, default=8192
        Mini-batch size for SGD training.
    n_policy_iterations : int, default=3
        Number of NPL-style policy iterations.
    verbose : bool, default=False
        Whether to print progress messages during estimation.

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
    ev_features_ : numpy.ndarray or None
        Per-feature EV component values of shape (n_states, n_features).
        Shows how much of the continuation value comes from each structural
        feature. Available after fitting if the underlying estimator
        includes them in metadata.
    converged_ : bool
        Whether the optimization converged.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    Examples
    --------
    >>> from econirl.estimators import TDCCP
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "bus_id": [0, 0, 1, 1],
    ...     "mileage": [10, 20, 15, 30],
    ...     "replaced": [0, 0, 0, 1],
    ... })
    >>>
    >>> model = TDCCP(n_states=90, hidden_dim=64, avi_iterations=20)
    >>> model.fit(df, state="mileage", action="replaced", id="bus_id")
    >>> print(model.params_)
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | RewardSpec = "linear_cost",
        se_method: Literal["robust", "asymptotic"] = "asymptotic",
        # Method selection (new: Adusumilli-Eckardt 2025)
        method: Literal["semigradient", "neural"] = "semigradient",
        cross_fitting: bool = True,
        robust_se: bool = True,
        # Semi-gradient specific
        basis_dim: int = 8,
        basis_type: Literal["polynomial", "encoded", "tabular"] = "polynomial",
        basis_include_rewards: bool = False,
        basis_ridge: float = 1e-8,
        basis_pinv_rcond: float | None = None,
        # Neural AVI specific
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        avi_iterations: int = 20,
        epochs_per_avi: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 8192,
        # CCP estimation
        ccp_method: Literal["frequency", "logit"] = "frequency",
        # NPL iteration (not in paper, optional)
        n_policy_iterations: int = 1,
        verbose: bool = False,
    ):
        """Initialize the TD-CCP estimator.

        Parameters
        ----------
        n_states : int, default=90
            Number of discrete states.
        n_actions : int, default=2
            Number of discrete actions.
        discount : float, default=0.9999
            Time discount factor (beta).
        utility : str or RewardSpec, default="linear_cost"
            Utility specification to use.
        se_method : str, default="asymptotic"
            Method for computing standard errors.
        method : str, default="semigradient"
            TD method: "semigradient" (fast closed-form, eq 3.5) or
            "neural" (AVI with neural networks, Algorithm 1).
        cross_fitting : bool, default=True
            Use 2-fold cross-fitting (Algorithm 2) for valid inference.
        robust_se : bool, default=True
            Compute locally robust standard errors (Section 4).
        basis_dim : int, default=8
            Number of polynomial basis functions for semi-gradient method.
        basis_type : str, default="polynomial"
            Semi-gradient basis: "polynomial", "encoded", or "tabular".
        basis_include_rewards : bool, default=False
            Include reward features in the encoded semi-gradient basis.
        basis_ridge : float, default=1e-8
            Ridge stabilization for the semi-gradient normal equation.
        basis_pinv_rcond : float, optional
            Pseudoinverse cutoff for nearly singular semi-gradient bases.
        hidden_dim : int, default=64
            Hidden units per layer in EV component networks.
        num_hidden_layers : int, default=2
            Number of hidden layers in EV component networks.
        avi_iterations : int, default=20
            Number of approximate value iteration rounds.
        epochs_per_avi : int, default=30
            Number of SGD epochs per AVI iteration.
        learning_rate : float, default=1e-3
            Learning rate for neural network training.
        batch_size : int, default=8192
            Mini-batch size for SGD training.
        ccp_method : str, default="frequency"
            CCP estimation: "frequency" or "logit".
        n_policy_iterations : int, default=1
            Number of NPL-style policy iterations. Paper uses 1.
        verbose : bool, default=False
            Whether to print progress messages.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.utility = utility
        self.se_method = se_method
        self.method = method
        self.cross_fitting = cross_fitting
        self.robust_se = robust_se
        self.basis_dim = basis_dim
        self.basis_type = basis_type
        self.basis_include_rewards = basis_include_rewards
        self.basis_ridge = basis_ridge
        self.basis_pinv_rcond = basis_pinv_rcond
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.avi_iterations = avi_iterations
        self.epochs_per_avi = epochs_per_avi
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ccp_method = ccp_method
        self.n_policy_iterations = n_policy_iterations
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
        self.ev_features_: np.ndarray | None = None

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
    ) -> "TDCCP":
        """Fit the TD-CCP estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with observations. When a DataFrame is passed,
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
            Reward/utility specification. If provided, overrides the
            ``utility`` parameter passed at construction time.

        Returns
        -------
        self : TDCCP
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

        # Create problem specification with state encoder for neural network
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
            state_dim=1,
            state_encoder=lambda s: jnp.expand_dims(jnp.asarray(s, dtype=jnp.float32) / max(self.n_states - 1, 1), axis=-1),
        )

        # Create the underlying TD-CCP estimator with all config options
        config = TDCCPConfig(
            method=self.method,
            basis_dim=self.basis_dim,
            basis_type=self.basis_type,
            basis_include_rewards=self.basis_include_rewards,
            basis_ridge=self.basis_ridge,
            basis_pinv_rcond=self.basis_pinv_rcond,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            avi_iterations=self.avi_iterations,
            epochs_per_avi=self.epochs_per_avi,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            ccp_method=self.ccp_method,
            cross_fitting=self.cross_fitting,
            robust_se=self.robust_se,
            n_policy_iterations=self.n_policy_iterations,
            compute_se=True,
            verbose=self.verbose,
        )
        estimator = TDCCPEstimator(config=config, se_method=self.se_method)

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
        """Build full transition tensor for both actions."""
        n = self.n_states
        transitions = np.zeros((self.n_actions, n, n), dtype=np.float32)
        transitions[0] = np.asarray(keep_transitions, dtype=np.float32)
        for s in range(n):
            transitions[1, s, :] = transitions[0, 0, :]
        return jnp.array(transitions)

    def _create_utility(self) -> LinearUtility:
        """Create utility function for estimation."""
        if self.utility != "linear_cost":
            raise ValueError(f"Unknown utility specification: {self.utility}")

        n = self.n_states
        features = jnp.zeros((n, self.n_actions, 2))
        mileage = jnp.arange(n, dtype=jnp.float32)
        features = features.at[:, 0, 0].set(-mileage)
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

        # Policy matrix
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

        # TD-CCP specific: EV feature components
        if self._result.metadata:
            ev = self._result.metadata.get("ev_features")
            if ev is not None:
                self.ev_features_ = np.asarray(ev)

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Structural reward matrix R(s,a) of shape (n_states, n_actions).

        Computes the utility matrix from the fitted parameters and the
        feature specification. Returns None if the model has not been fitted.
        """
        if self.params_ is None or self._utility_fn is None or self._result is None:
            return None
        param_names = self._result.parameter_names
        param_vector = jnp.array(
            [self.params_[name] for name in param_names],
            dtype=jnp.float32,
        )
        utility_matrix = self._utility_fn.compute(param_vector)
        return np.asarray(utility_matrix)

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "TD-CCP: Not fitted yet. Call fit() first."

        return self._result.summary()

    def conf_int(self, alpha: float = 0.05) -> dict:
        """Compute confidence intervals for parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Returns (1 - alpha) confidence intervals.

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
            Each row sums to 1.
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)

        # Get policy (choice probabilities) from result
        policy = np.asarray(self._result.policy)

        # Index into the policy for the requested states
        proba = policy[states]

        return proba

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"TDCCP(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, fitted=True)"
            )
        return (
            f"TDCCP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted=False)"
        )
