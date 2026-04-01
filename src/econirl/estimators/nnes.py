"""Sklearn-style NNES estimator for dynamic discrete choice models.

This module provides an NNES class with a scikit-learn style API that wraps
the underlying NNESEstimator from econirl.estimation.nnes. It accepts pandas
DataFrames with column names instead of the low-level Panel API.

NNES (Neural Network Estimation of Structural models, Nguyen 2025) trains
a neural V(s) network in Phase 1 and uses it in a structural MLE in Phase 2.
Two Bellman variants are available via the ``bellman`` parameter:

- ``"npl"`` (default): Uses the NPL Bellman with Hotz-Miller emax correction.
  Has the zero Jacobian property (Neyman orthogonality), so standard errors
  are semiparametrically efficient despite V-approximation error.
- ``"nfxp"``: Uses the NFXP soft Bellman operator. Does NOT have Neyman
  orthogonality. V-approximation errors contaminate the score.

Example:
    >>> from econirl.estimators import NNES
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>> model = NNES(n_states=90, discount=0.9999, bellman="npl")
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>> print(model.params_)
    >>> print(model.summary())
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimation.nnes import NNESConfig, NNESEstimator, NNESNFXPEstimator
from econirl.preferences.linear import LinearUtility
from econirl.transitions import TransitionEstimator


class NNES:
    """Sklearn-style NNES estimator for dynamic discrete choice models.

    NNES (Neural Network Estimation of Structural models) estimates utility
    parameters using a two-phase approach: Phase 1 trains a neural V(s)
    network, Phase 2 uses the learned V-network in a structural MLE for
    theta (Nguyen 2025).

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
    bellman : str, default="npl"
        Which Bellman equation to use in Phase 1.  ``"npl"`` uses the NPL
        Bellman with Hotz-Miller emax correction (has Neyman orthogonality,
        standard errors are semiparametrically efficient).  ``"nfxp"`` uses
        the NFXP soft Bellman operator (no orthogonality, V-errors
        contaminate the score).
    se_method : str, default="asymptotic"
        Method for computing standard errors. Options: "robust", "asymptotic".
    hidden_dim : int, default=32
        Number of hidden units per layer in the V-network.
    num_layers : int, default=2
        Number of hidden layers in the V-network.
    v_lr : float, default=1e-3
        Learning rate for V-network training (Phase 1).
    v_epochs : int, default=500
        Number of training epochs for V-network per outer iteration.
    n_outer_iterations : int, default=3
        Number of Phase 1 + Phase 2 alternations.
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
    v_network_ : numpy.ndarray or None
        V-network values for all states after training, shape (n_states,).
    converged_ : bool
        Whether the optimization converged.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    Examples
    --------
    >>> from econirl.estimators import NNES
    >>> model = NNES(n_states=90, bellman="npl")  # default: NPL Bellman
    >>> model.fit(df, state="mileage", action="replaced", id="bus_id")
    >>> print(model.params_)
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | RewardSpec = "linear_cost",
        bellman: Literal["npl", "nfxp"] = "npl",
        se_method: Literal["robust", "asymptotic"] = "asymptotic",
        hidden_dim: int = 32,
        num_layers: int = 2,
        v_lr: float = 1e-3,
        v_epochs: int = 500,
        n_outer_iterations: int = 3,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.utility = utility
        self.bellman = bellman
        self.se_method = se_method
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.v_lr = v_lr
        self.v_epochs = v_epochs
        self.n_outer_iterations = n_outer_iterations
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
        self.v_network_: np.ndarray | None = None

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
    ) -> "NNES":
        """Fit the NNES estimator to data.

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
        self : NNES
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

        # Create the underlying NNES estimator (NPL or NFXP variant)
        config = NNESConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            v_lr=self.v_lr,
            v_epochs=self.v_epochs,
            n_outer_iterations=self.n_outer_iterations,
            compute_se=True,
            se_method=self.se_method,
            verbose=self.verbose,
        )
        if self.bellman == "nfxp":
            estimator = NNESNFXPEstimator(config=config)
        else:
            estimator = NNESEstimator(config=config)

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
        transitions[0] = np.asarray(keep_transitions, dtype=np.float32)

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

        # Build feature matrix for linear cost utility
        # U(s, keep) = -theta_c * s
        # U(s, replace) = -RC
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

        # NNES-specific: V-network values
        if self._result.metadata:
            v_vals = self._result.metadata.get("v_network_values")
            if v_vals is not None:
                self.v_network_ = np.asarray(v_vals)

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "NNES: Not fitted yet. Call fit() first."

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
        fitted = self.params_ is not None
        return (
            f"NNES(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, bellman={self.bellman!r}, fitted={fitted})"
        )
