"""Sklearn-style MCE IRL estimator.

Maximum Causal Entropy Inverse Reinforcement Learning with sklearn-style API.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import norm as scipy_norm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.reward import LinearReward
from econirl.transitions import TransitionEstimator


class MCEIRL:
    """Sklearn-style Maximum Causal Entropy IRL estimator.

    Maximum Causal Entropy IRL (Ziebart 2010) recovers reward function
    parameters from demonstrated behavior, properly accounting for the
    causal structure of sequential decisions.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states.
    n_actions : int, default=2
        Number of discrete actions.
    discount : float, default=0.99
        Time discount factor (beta). Use <0.999 for numerical stability.
    feature_matrix : numpy.ndarray, optional
        State feature matrix of shape (n_states, n_features).
        If None, uses state index as single feature.
    feature_names : list[str], optional
        Names for each feature.
    se_method : str, default="bootstrap"
        Method for standard errors: "bootstrap", "asymptotic", or "hessian".
    n_bootstrap : int, default=100
        Number of bootstrap samples for SE computation.
    verbose : bool, default=False
        Print progress messages.

    Attributes
    ----------
    params_ : dict
        Estimated reward parameters {name: value}.
    se_ : dict
        Standard errors for each parameter.
    coef_ : numpy.ndarray
        Coefficients as array.
    reward_ : numpy.ndarray
        Recovered reward R(s) for each state.
    policy_ : numpy.ndarray
        Learned policy π(a|s), shape (n_states, n_actions).
    value_function_ : numpy.ndarray
        Value function V(s) for each state.
    state_visitation_ : numpy.ndarray
        Expected state visitation frequencies.
    log_likelihood_ : float
        Log-likelihood of the data under learned model.
    converged_ : bool
        Whether optimization converged.

    Examples
    --------
    >>> from econirl.estimators import MCEIRL
    >>> from econirl.datasets import load_rust_bus
    >>>
    >>> df = load_rust_bus()
    >>>
    >>> # State features: linear and quadratic mileage cost
    >>> n_states = 90
    >>> s = np.arange(n_states)
    >>> features = np.column_stack([s / 100, (s / 100) ** 2])
    >>>
    >>> model = MCEIRL(
    ...     n_states=n_states,
    ...     discount=0.99,
    ...     feature_matrix=features,
    ...     feature_names=["linear", "quadratic"],
    ...     verbose=True,
    ... )
    >>> model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
    >>> print(model.summary())

    References
    ----------
    Ziebart, B. D. (2010). Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy. PhD thesis, CMU.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        feature_matrix: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        se_method: Literal["bootstrap", "asymptotic", "hessian"] = "bootstrap",
        n_bootstrap: int = 100,
        inner_max_iter: int = 10000,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.feature_matrix = feature_matrix
        self.feature_names = feature_names
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.inner_max_iter = inner_max_iter
        self.verbose = verbose

        # Fitted attributes
        self.params_: dict | None = None
        self.se_: dict | None = None
        self.pvalues_: dict | None = None
        self.coef_: np.ndarray | None = None
        self.reward_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_function_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.state_visitation_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.converged_: bool | None = None
        self.reward_spec_: RewardSpec | None = None

        # Internal
        self._result = None
        self._panel = None
        self._reward_fn = None
        self._problem = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        reward: RewardSpec | None = None,
    ) -> "MCEIRL":
        """Fit the MCE IRL estimator.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with demonstrations.  When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
            When a Panel/TrajectoryPanel is passed, column names are ignored.
        state : str, optional
            Column name for state variable (required for DataFrame input).
        action : str, optional
            Column name for action variable (required for DataFrame input).
        id : str, optional
            Column name for individual/trajectory identifier (required for
            DataFrame input).
        transitions : numpy.ndarray, optional
            Pre-estimated transition matrix (n_states, n_states).
            If None, estimated from data.
        reward : RewardSpec, optional
            Reward/utility specification.  If provided, overrides the
            ``feature_matrix`` and ``feature_names`` parameters passed at
            construction time.

        Returns
        -------
        self : MCEIRL
            Fitted estimator.
        """
        # --- Handle reward spec ---
        if reward is not None:
            self.reward_spec_ = reward

        # --- Handle data: DataFrame or Panel/TrajectoryPanel ---
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            self._panel = self._dataframe_to_panel(data, state, action, id)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, "
                f"got {type(data)}"
            )

        # Estimate transitions
        if transitions is None:
            trans_est = TransitionEstimator(n_states=self.n_states, max_increase=2)
            trans_est.fit(self._panel)
            self.transitions_ = trans_est.matrix_
        else:
            self.transitions_ = np.asarray(transitions)

        # Build transition tensor
        transition_tensor = self._build_transition_tensor(self.transitions_)

        # Create problem
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        # Create reward function (RewardSpec overrides feature_matrix)
        if self.reward_spec_ is not None:
            self._reward_fn = self.reward_spec_.to_linear_reward()
        else:
            self._reward_fn = self._create_reward()

        # Create estimator with config
        config = MCEIRLConfig(
            se_method=self.se_method,
            n_bootstrap=self.n_bootstrap,
            inner_max_iter=self.inner_max_iter,
            verbose=self.verbose,
        )
        estimator = MCEIRLEstimator(config=config)

        # Estimate
        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._reward_fn,
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
        """Convert DataFrame to Panel."""
        trajectories = []

        for ind_id, group in data.groupby(id, sort=True):
            sorted_group = group.sort_index()

            states = sorted_group[state].values.astype(np.int64)
            actions = sorted_group[action].values.astype(np.int64)

            # Compute next states
            next_states = np.zeros_like(states)
            next_states[:-1] = states[1:]
            if len(states) > 0:
                last_action = actions[-1]
                if last_action == 1:
                    next_states[-1] = 0
                else:
                    next_states[-1] = min(states[-1] + 1, self.n_states - 1)

            traj = Trajectory(
                states=np.array(states, dtype=np.int64),
                actions=np.array(actions, dtype=np.int64),
                next_states=np.array(next_states, dtype=np.int64),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _build_transition_tensor(self, keep_transitions: np.ndarray) -> jnp.ndarray:
        """Build transition tensor for both actions."""
        n = self.n_states
        transitions = np.zeros((self.n_actions, n, n), dtype=np.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = np.asarray(keep_transitions, dtype=np.float32)

        # Action 1 (replace): reset to state 0, then transition
        for s in range(n):
            transitions[1, s, :] = transitions[0, 0, :]

        return jnp.array(transitions)

    def _create_reward(self) -> LinearReward:
        """Create reward function."""
        if self.feature_matrix is not None:
            features = jnp.array(self.feature_matrix, dtype=jnp.float32)
            n_features = features.shape[1]
        else:
            features = jnp.expand_dims(jnp.arange(self.n_states, dtype=jnp.float32), axis=1)
            n_features = 1

        if self.feature_names is not None:
            param_names = list(self.feature_names)
        else:
            param_names = [f"f{i}" for i in range(n_features)]

        return LinearReward(
            state_features=features,
            parameter_names=param_names,
            n_actions=self.n_actions,
        )

    def _extract_results(self) -> None:
        """Extract results into sklearn-style attributes."""
        if self._result is None:
            return

        params = np.asarray(self._result.parameters)
        param_names = self._result.parameter_names

        self.params_ = {name: float(val) for name, val in zip(param_names, params)}
        self.coef_ = params.copy()

        # Standard errors from metadata
        if self._result.metadata and "standard_errors" in self._result.metadata:
            se_values = self._result.metadata["standard_errors"]
            if se_values is not None:
                self.se_ = {name: float(val) for name, val in zip(param_names, se_values)}
            else:
                se = np.asarray(self._result.standard_errors)
                self.se_ = {name: float(val) for name, val in zip(param_names, se)}
        else:
            se = np.asarray(self._result.standard_errors)
            self.se_ = {name: float(val) for name, val in zip(param_names, se)}

        # P-values from t-statistics (Wald test)
        if self.se_ is not None:
            pvalues: dict[str, float] = {}
            for name in self.params_:
                se_val = self.se_[name]
                if se_val and se_val > 0 and np.isfinite(se_val):
                    t_stat = self.params_[name] / se_val
                    pvalues[name] = float(
                        2 * (1 - scipy_norm.cdf(abs(t_stat)))
                    )
                else:
                    pvalues[name] = float("nan")
            self.pvalues_ = pvalues

        # Reward function R(s)
        reward_params = jnp.array(params, dtype=jnp.float32)
        reward_matrix = self._reward_fn.compute(reward_params)
        self.reward_ = np.asarray(reward_matrix[:, 0])

        # Policy
        if self._result.policy is not None:
            self.policy_ = np.asarray(self._result.policy)

        # Value function
        if self._result.value_function is not None:
            self.value_function_ = np.asarray(self._result.value_function)
            self.value_ = self.value_function_

        # State visitation
        if self._result.metadata and "state_visitation" in self._result.metadata:
            self.state_visitation_ = np.array(self._result.metadata["state_visitation"])

        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Predict choice probabilities.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.

        Returns
        -------
        proba : numpy.ndarray
            Choice probabilities, shape (len(states), n_actions).
        """
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)
        return self.policy_[states]

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
        """Generate formatted summary of results."""
        if self._result is None:
            return "MCEIRL: Not fitted yet. Call fit() first."

        lines = []
        lines.append("=" * 70)
        lines.append("Maximum Causal Entropy IRL Results".center(70))
        lines.append("=" * 70)
        lines.append(f"{'Method:':<25} MCE IRL (Ziebart 2010)")
        lines.append(f"{'Discount Factor (β):':<25} {self.discount}")
        lines.append(f"{'No. States:':<25} {self.n_states}")
        lines.append(f"{'No. Actions:':<25} {self.n_actions}")
        lines.append(f"{'Log-Likelihood:':<25} {self.log_likelihood_:,.2f}")
        lines.append(f"{'Converged:':<25} {'Yes' if self.converged_ else 'No'}")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Parameter Estimates:")
        lines.append("-" * 70)
        lines.append(f"{'Parameter':<20} {'Estimate':>12} {'Std Err':>12} {'t-stat':>10} {'95% CI':>20}")
        lines.append("-" * 70)

        for name in self.params_:
            param = self.params_[name]
            se = self.se_.get(name, float('nan')) if self.se_ else float('nan')

            if np.isfinite(se) and se > 0:
                t_stat = param / se
                ci_low = param - 1.96 * se
                ci_high = param + 1.96 * se
                ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
            else:
                t_stat = float('nan')
                ci_str = "[nan, nan]"

            lines.append(f"{name:<20} {param:>12.4f} {se:>12.4f} {t_stat:>10.2f} {ci_str:>20}")

        lines.append("-" * 70)

        # Feature matching diagnostics
        if self._result.metadata:
            emp = self._result.metadata.get("empirical_features", [])
            exp = self._result.metadata.get("final_expected_features", [])
            diff = self._result.metadata.get("feature_difference", 0)

            lines.append("")
            lines.append("Feature Matching Diagnostics:")
            lines.append(f"  Feature difference (||μ_D - μ_π||): {diff:.6f}")
            if emp and exp:
                lines.append(f"  Empirical features: {[f'{x:.4f}' for x in emp]}")
                lines.append(f"  Expected features:  {[f'{x:.4f}' for x in exp]}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = self.params_ is not None
        return (
            f"MCEIRL(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted={fitted})"
        )
