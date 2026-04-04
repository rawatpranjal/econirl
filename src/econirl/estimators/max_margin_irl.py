"""Sklearn-style Max Margin IRL estimator.

This module provides a MaxMarginIRL class with a scikit-learn style API that wraps
the underlying MaxMarginIRLEstimator from econirl.estimation.max_margin_irl. It
implements the Abbeel & Ng (2004) apprenticeship learning algorithm.

Example:
    >>> from econirl.estimators import MaxMarginIRL
    >>> import pandas as pd
    >>>
    >>> # Load expert demonstration data
    >>> df = pd.read_csv("expert_demos.csv")
    >>>
    >>> # Create estimator and fit
    >>> model = MaxMarginIRL(n_states=90, n_actions=2, discount=0.99)
    >>> model.fit(data=df, state="state", action="action", id="agent_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"feature_0": 0.5, "feature_1": -0.3, ...}
    >>> print(model.reward_)        # R(s) array for each state
    >>> print(model.summary())
    >>>
    >>> # Predict choice probabilities
    >>> proba = model.predict_proba(states=np.array([0, 10, 50]))
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator
from econirl.preferences.reward import LinearReward
from econirl.transitions import TransitionEstimator


class MaxMarginIRL:
    """Sklearn-style Max Margin IRL estimator (Abbeel & Ng 2004).

    Maximum Margin Inverse Reinforcement Learning finds reward weights that
    make the expert policy better than any other policy by a margin. It uses
    an iterative constraint generation approach to solve a quadratic program.

    This is useful when you have expert demonstrations and want to recover
    the underlying reward function that explains the behavior.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states.
    n_actions : int, default=2
        Number of discrete actions.
    discount : float, default=0.99
        Time discount factor (gamma).
    n_features : int, default=None
        Number of reward features. If None, uses n_states (one-hot encoding).
    features : numpy.ndarray, optional
        State feature matrix of shape (n_states, n_features). If None,
        uses identity matrix (one-hot state features).
    feature_names : list[str], optional
        Names for each feature. If None, uses "feature_0", "feature_1", etc.
    max_iterations : int, default=50
        Maximum constraint generation iterations.
    margin_tol : float, default=1e-4
        Convergence tolerance on margin improvement.
    se_method : str, default="asymptotic"
        Method for computing standard errors.
    verbose : bool, default=False
        Whether to print progress messages during estimation.

    Attributes
    ----------
    params_ : dict
        Estimated reward weights after fitting. Keys are feature names
        and values are point estimates.
    se_ : dict
        Standard errors for each parameter.
    coef_ : numpy.ndarray
        Coefficients as a numpy array (sklearn convention).
    reward_ : numpy.ndarray
        Recovered reward R(s) for each state, shape (n_states,).
    margin_ : float
        Achieved margin between expert and best alternative policy.
    value_function_ : numpy.ndarray
        Estimated value function V(s) for each state.
    transitions_ : numpy.ndarray
        Transition probability matrix (n_states x n_states).
    converged_ : bool
        Whether the optimization converged.

    Examples
    --------
    >>> from econirl.estimators import MaxMarginIRL
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "agent_id": [0, 0, 1, 1],
    ...     "state": [10, 20, 15, 30],
    ...     "action": [0, 0, 0, 1],
    ... })
    >>>
    >>> model = MaxMarginIRL(n_states=90, n_actions=2)
    >>> model.fit(df, state="state", action="action", id="agent_id")
    >>> print(model.reward_)  # Recovered reward for each state

    References
    ----------
    Abbeel, P., & Ng, A. Y. (2004). "Apprenticeship learning via inverse
        reinforcement learning." In Proceedings of ICML.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        n_features: int | None = None,
        features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        max_iterations: int = 50,
        margin_tol: float = 1e-4,
        se_method: Literal["asymptotic"] = "asymptotic",
        verbose: bool = False,
    ):
        """Initialize the MaxMarginIRL estimator.

        Parameters
        ----------
        n_states : int, default=90
            Number of discrete states.
        n_actions : int, default=2
            Number of discrete actions.
        discount : float, default=0.99
            Time discount factor.
        n_features : int, optional
            Number of reward features. If None, defaults to n_states.
        features : numpy.ndarray, optional
            State feature matrix (n_states, n_features). If None, uses identity.
        feature_names : list[str], optional
            Names for each feature.
        max_iterations : int, default=50
            Maximum constraint generation iterations.
        margin_tol : float, default=1e-4
            Convergence tolerance on margin.
        se_method : str, default="asymptotic"
            Method for computing standard errors.
        verbose : bool, default=False
            Whether to print progress messages.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.n_features = n_features if n_features is not None else n_states
        self.features = features
        self.feature_names = feature_names
        self.max_iterations = max_iterations
        self.margin_tol = margin_tol
        self.se_method = se_method
        self.verbose = verbose

        # Fitted attributes (set after fit())
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.reward_: np.ndarray | None = None
        self.margin_: float | None = None
        self.value_function_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.converged_: bool | None = None

        # Internal storage
        self._result = None
        self._panel = None
        self._reward_fn = None
        self._problem = None

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        transitions: np.ndarray | None = None,
    ) -> "MaxMarginIRL":
        """Fit the MaxMarginIRL estimator to expert demonstration data.

        Parameters
        ----------
        data : pandas.DataFrame
            Panel data with expert demonstrations. Must contain columns for
            state, action, and individual id.
        state : str
            Column name for the state variable.
        action : str
            Column name for the action variable.
        id : str
            Column name for the individual/agent identifier.
        transitions : numpy.ndarray, optional
            Pre-estimated transition matrix of shape (n_states, n_states).
            If None, transitions are estimated from the data.

        Returns
        -------
        self : MaxMarginIRL
            Returns self for method chaining.
        """
        # Convert DataFrame to Panel
        self._panel = self._dataframe_to_panel(data, state, action, id)

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

        # Create reward function with state features
        self._reward_fn = self._create_reward()

        # Create the underlying MaxMarginIRL estimator
        estimator = MaxMarginIRLEstimator(
            se_method=self.se_method,
            max_iterations=self.max_iterations,
            margin_tol=self.margin_tol,
            verbose=self.verbose,
        )

        # Run estimation
        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._reward_fn,  # LinearReward is a UtilityFunction
            problem=self._problem,
            transitions=transition_tensor,
        )

        # Extract results
        self._extract_results()

        return self

    def _dataframe_to_panel(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
    ) -> Panel:
        """Convert pandas DataFrame to Panel object.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.
        state : str
            Column name for state.
        action : str
            Column name for action.
        id : str
            Column name for individual id.

        Returns
        -------
        Panel
            Panel object suitable for estimation.
        """
        trajectories = []

        # Group by individual
        for ind_id, group in data.groupby(id, sort=True):
            # Sort by any time/period column if available, otherwise keep order
            sorted_group = group.sort_index()

            states = sorted_group[state].values.astype(np.int64)
            actions = sorted_group[action].values.astype(np.int64)

            # Compute next states
            next_states = np.zeros_like(states)
            next_states[:-1] = states[1:]
            # For the last observation, apply transition logic based on action
            if len(states) > 0:
                last_state = states[-1]
                last_action = actions[-1]
                if last_action == 1:  # Replace -> reset to 0
                    next_states[-1] = 0
                else:  # Keep -> stay at same state (conservative)
                    next_states[-1] = min(last_state + 1, self.n_states - 1)

            traj = Trajectory(
                states=np.array(states, dtype=np.int64),
                actions=np.array(actions, dtype=np.int64),
                next_states=np.array(next_states, dtype=np.int64),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _build_transition_tensor(self, keep_transitions: np.ndarray) -> np.ndarray:
        """Build full transition tensor for both actions.

        Parameters
        ----------
        keep_transitions : numpy.ndarray
            Transition matrix for action=0 (keep), shape (n_states, n_states).

        Returns
        -------
        numpy.ndarray
            Transition tensor of shape (n_actions, n_states, n_states).
        """
        n = self.n_states
        transitions = np.zeros((self.n_actions, n, n), dtype=np.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = np.array(keep_transitions, dtype=np.float32)

        # Action 1 (replace): reset to state 0, then transition
        # After replacement, start at state 0 and apply the same transition
        for s in range(n):
            # After replacement from any state, we transition as if from state 0
            transitions[1, s, :] = transitions[0, 0, :]

        return transitions

    def _create_reward(self) -> LinearReward:
        """Create reward function for estimation.

        Returns
        -------
        LinearReward
            Reward function with appropriate state features.
        """
        # Build feature matrix
        if self.features is not None:
            # Use provided features
            feature_matrix = np.array(self.features, dtype=np.float32)
            if feature_matrix.shape[0] != self.n_states:
                raise ValueError(
                    f"features must have {self.n_states} rows, "
                    f"got {feature_matrix.shape[0]}"
                )
            n_features = feature_matrix.shape[1]
        else:
            # Use identity (one-hot state encoding) if n_features == n_states
            # Otherwise use random features (for dimensionality reduction)
            if self.n_features == self.n_states:
                feature_matrix = np.eye(self.n_states, dtype=np.float32)
            else:
                # Use simple state-dependent features
                # e.g., polynomial features of normalized state
                feature_matrix = np.zeros(
                    (self.n_states, self.n_features), dtype=np.float32
                )
                states_normalized = np.arange(
                    self.n_states, dtype=np.float32
                ) / (self.n_states - 1)
                for k in range(self.n_features):
                    # Polynomial features: s^0, s^1, s^2, ...
                    feature_matrix[:, k] = states_normalized ** k

            n_features = self.n_features

        # Create parameter names
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise ValueError(
                    f"feature_names must have {n_features} elements, "
                    f"got {len(self.feature_names)}"
                )
            param_names = list(self.feature_names)
        else:
            param_names = [f"feature_{i}" for i in range(n_features)]

        return LinearReward(
            state_features=feature_matrix,
            parameter_names=param_names,
            n_actions=self.n_actions,
        )

    def _extract_results(self) -> None:
        """Extract results from estimation into sklearn-style attributes."""
        if self._result is None:
            return

        # Parameter estimates (reward weights)
        params = np.asarray(self._result.parameters)
        param_names = self._result.parameter_names

        self.params_ = {name: float(val) for name, val in zip(param_names, params)}
        self.coef_ = params.copy()

        # Standard errors
        se = np.asarray(self._result.standard_errors)
        self.se_ = {name: float(val) for name, val in zip(param_names, se)}

        # Compute reward for each state: R(s) = theta * phi(s)
        theta = self._result.parameters
        reward_matrix = self._reward_fn.compute(theta)  # (n_states, n_actions)
        self.reward_ = np.asarray(reward_matrix[:, 0])  # Same for all actions

        # Other attributes
        self.converged_ = bool(self._result.converged)

        # Get margin from metadata if available
        if self._result.metadata and "margin" in self._result.metadata:
            self.margin_ = float(self._result.metadata["margin"])

        if self._result.value_function is not None:
            self.value_function_ = np.asarray(self._result.value_function)

        if self._result.policy is not None:
            self.policy_ = np.asarray(self._result.policy)

        # p-values from t-statistics
        if self.se_ is not None and self.params_ is not None:
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

    @property
    def value_(self) -> np.ndarray | None:
        """Value function V(s) of shape (n_states,)."""
        return self.value_function_

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Reward matrix R(s,a) of shape (n_states, n_actions).

        MaxMarginIRL learns a state-only reward R(s). This property broadcasts
        it to all actions so that the shape matches the protocol requirement.
        """
        if self.reward_ is None:
            return None
        return np.tile(self.reward_[:, np.newaxis], (1, self.n_actions))

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

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "MaxMarginIRL: Not fitted yet. Call fit() first."

        return self._result.summary()

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
                f"MaxMarginIRL(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, fitted=True)"
            )
        return (
            f"MaxMarginIRL(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted=False)"
        )
