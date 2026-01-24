"""Sklearn-style NFXP estimator for dynamic discrete choice models.

This module provides an NFXP class with a scikit-learn style API that wraps
the underlying NFXPEstimator from econirl.estimation.nfxp. It accepts pandas
DataFrames with column names instead of the low-level Panel API.

Example:
    >>> from econirl.estimators import NFXP
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit
    >>> model = NFXP(n_states=90, discount=0.9999, utility="linear_cost")
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"theta_c": 0.001, "RC": 9.35}
    >>> print(model.coef_)          # numpy array [0.001, 9.35]
    >>> print(model.log_likelihood_)
    >>> print(model.summary())
    >>>
    >>> # Predict choice probabilities
    >>> proba = model.predict_proba(states=np.array([0, 10, 50]))
    >>>
    >>> # Simulate new data under estimated policy
    >>> sim_df = model.simulate(n_agents=100, n_periods=50, seed=42)
    >>>
    >>> # Counterfactual analysis: what if RC was higher?
    >>> cf_result = model.counterfactual(RC=15.0)
    >>> print(cf_result.policy)  # New policy under higher replacement cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility
from econirl.transitions import TransitionEstimator


@dataclass
class CounterfactualResult:
    """Result of a counterfactual policy analysis.

    Contains the value function and policy computed under alternative
    parameter values, enabling "what if" analysis after estimation.

    Attributes:
        params: Dictionary of parameter values used in the counterfactual.
                Contains both the modified parameters and the original
                estimated values for unchanged parameters.
        value_function: Value function V(s) under the counterfactual parameters,
                       shape (n_states,).
        policy: Choice probabilities P(a|s) under the counterfactual parameters,
               shape (n_states, n_actions). Each row sums to 1.

    Example:
        >>> # After fitting NFXP estimator
        >>> cf = estimator.counterfactual(RC=15.0)
        >>> print(f"Under RC=15.0, P(replace|state=50) = {cf.policy[50, 1]:.3f}")
    """

    params: dict[str, float]
    value_function: np.ndarray
    policy: np.ndarray


class NFXP:
    """Sklearn-style NFXP estimator for dynamic discrete choice models.

    The Nested Fixed Point (NFXP) algorithm estimates utility parameters by
    nesting the solution of the Bellman equation within likelihood maximization.
    This is the classic approach from Rust (1987, 1988).

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states (e.g., mileage bins).
    n_actions : int, default=2
        Number of discrete actions (e.g., keep/replace).
    discount : float, default=0.9999
        Time discount factor (beta).
    utility : str, default="linear_cost"
        Utility specification. Currently supports "linear_cost" which
        implements the Rust bus model: u = -theta_c * s * (1-a) - RC * a
    se_method : str, default="robust"
        Method for computing standard errors. Options: "robust", "asymptotic".
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
    value_function_ : numpy.ndarray
        Estimated value function V(s) for each state.
    transitions_ : numpy.ndarray
        Transition probability matrix (n_states x n_states).
    converged_ : bool
        Whether the optimization converged.

    Examples
    --------
    >>> from econirl.estimators import NFXP
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "bus_id": [0, 0, 1, 1],
    ...     "mileage": [10, 20, 15, 30],
    ...     "replaced": [0, 0, 0, 1],
    ... })
    >>>
    >>> model = NFXP(n_states=90)
    >>> model.fit(df, state="mileage", action="replaced", id="bus_id")
    >>> print(model.params_)
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: Literal["linear_cost"] = "linear_cost",
        se_method: Literal["robust", "asymptotic"] = "robust",
        verbose: bool = False,
    ):
        """Initialize the NFXP estimator.

        Parameters
        ----------
        n_states : int, default=90
            Number of discrete states.
        n_actions : int, default=2
            Number of discrete actions.
        discount : float, default=0.9999
            Time discount factor (beta).
        utility : str, default="linear_cost"
            Utility specification to use.
        se_method : str, default="robust"
            Method for computing standard errors.
        verbose : bool, default=False
            Whether to print progress messages.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.utility = utility
        self.se_method = se_method
        self.verbose = verbose

        # Fitted attributes (set after fit())
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.value_function_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.converged_: bool | None = None

        # Internal storage
        self._result = None
        self._panel = None
        self._utility_fn = None
        self._problem = None

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        transitions: np.ndarray | None = None,
    ) -> "NFXP":
        """Fit the NFXP estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame
            Panel data with observations. Must contain columns for state,
            action, and individual id.
        state : str
            Column name for the state variable.
        action : str
            Column name for the action variable.
        id : str
            Column name for the individual identifier.
        transitions : numpy.ndarray, optional
            Pre-estimated transition matrix of shape (n_states, n_states).
            If None, transitions are estimated from the data.

        Returns
        -------
        self : NFXP
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
        # Action 0 (keep): use estimated transitions
        # Action 1 (replace): reset to state 0, then apply transition
        transition_tensor = self._build_transition_tensor(self.transitions_)

        # Create problem specification
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        # Create utility function
        self._utility_fn = self._create_utility()

        # Create the underlying NFXP estimator
        estimator = NFXPEstimator(
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
            # For the last observation, we need to handle it specially
            # We'll use the "wrap around" convention or just use the current state
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
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _build_transition_tensor(self, keep_transitions: np.ndarray) -> torch.Tensor:
        """Build full transition tensor for both actions.

        Parameters
        ----------
        keep_transitions : numpy.ndarray
            Transition matrix for action=0 (keep), shape (n_states, n_states).

        Returns
        -------
        torch.Tensor
            Transition tensor of shape (n_actions, n_states, n_states).
        """
        n = self.n_states
        transitions = torch.zeros((self.n_actions, n, n), dtype=torch.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = torch.tensor(keep_transitions, dtype=torch.float32)

        # Action 1 (replace): reset to state 0, then transition
        # After replacement, start at state 0 and apply the same transition
        # The row for state 0 in keep_transitions tells us where we go from 0
        for s in range(n):
            # After replacement from any state, we transition as if from state 0
            transitions[1, s, :] = transitions[0, 0, :]

        return transitions

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
        features = torch.zeros((n, self.n_actions, 2), dtype=torch.float32)

        mileage = torch.arange(n, dtype=torch.float32)

        # Keep action (a=0): feature = [-s, 0]
        features[:, 0, 0] = -mileage
        features[:, 0, 1] = 0.0

        # Replace action (a=1): feature = [0, -1]
        features[:, 1, 0] = 0.0
        features[:, 1, 1] = -1.0

        return LinearUtility(
            feature_matrix=features,
            parameter_names=["theta_c", "RC"],
        )

    def _extract_results(self) -> None:
        """Extract results from estimation into sklearn-style attributes."""
        if self._result is None:
            return

        # Parameter estimates
        params = self._result.parameters.numpy()
        param_names = self._result.parameter_names

        self.params_ = {name: float(val) for name, val in zip(param_names, params)}
        self.coef_ = params.copy()

        # Standard errors
        se = self._result.standard_errors.numpy()
        self.se_ = {name: float(val) for name, val in zip(param_names, se)}

        # Other attributes
        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)

        if self._result.value_function is not None:
            self.value_function_ = self._result.value_function.numpy()

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "NFXP: Not fitted yet. Call fit() first."

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
        policy = self._result.policy.numpy()

        # Index into the policy for the requested states
        proba = policy[states]

        return proba

    def simulate(
        self,
        n_agents: int,
        n_periods: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Simulate choices under the estimated policy.

        Generates synthetic data by simulating agents making decisions
        according to the fitted model. Each agent starts at state 0 and
        evolves according to the estimated transition probabilities and
        choice probabilities.

        Parameters
        ----------
        n_agents : int
            Number of agents to simulate.
        n_periods : int
            Number of time periods per agent.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - agent_id: Identifier for each agent (0 to n_agents-1)
            - period: Time period (0 to n_periods-1)
            - state: State at the beginning of the period
            - action: Action taken (sampled from estimated policy)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = NFXP(n_states=90)
        >>> model.fit(data, state="mileage", action="replaced", id="bus_id")
        >>> sim_data = model.simulate(n_agents=100, n_periods=50, seed=42)
        >>> print(sim_data.head())
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Get the policy (choice probabilities)
        policy = self._result.policy.numpy()  # shape: (n_states, n_actions)

        # Get transition probabilities (for action 0 = keep)
        transitions = self.transitions_  # shape: (n_states, n_states)

        # Storage for results
        data = []

        for agent_id in range(n_agents):
            state = 0  # All agents start at state 0

            for period in range(n_periods):
                # Sample action from policy
                action_probs = policy[state]
                action = np.random.choice(self.n_actions, p=action_probs)

                # Record observation
                data.append({
                    "agent_id": agent_id,
                    "period": period,
                    "state": state,
                    "action": action,
                })

                # Transition to next state
                if action == 1:  # Replace: reset to state 0, then transition
                    state = 0
                    trans_probs = transitions[0]
                else:  # Keep: use transition from current state
                    trans_probs = transitions[state]

                state = np.random.choice(self.n_states, p=trans_probs)

        return pd.DataFrame(data)

    def counterfactual(self, **param_changes) -> CounterfactualResult:
        """Compute outcomes under different parameter values.

        Performs counterfactual analysis by solving the dynamic programming
        problem under alternative parameter values. This enables "what if"
        questions like "what would the policy be if RC was 15 instead of 10?"

        Parameters
        ----------
        **param_changes : float
            Keyword arguments specifying parameter changes.
            Keys must be valid parameter names (e.g., "theta_c", "RC").
            Values are the counterfactual parameter values.

        Returns
        -------
        CounterfactualResult
            Object containing:
            - params: Dictionary of all parameter values used
            - value_function: V(s) under new parameters
            - policy: P(a|s) under new parameters

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If an unknown parameter name is provided.

        Examples
        --------
        >>> model = NFXP(n_states=90)
        >>> model.fit(data, state="mileage", action="replaced", id="bus_id")
        >>>
        >>> # What if replacement cost was higher?
        >>> cf = model.counterfactual(RC=15.0)
        >>> print(f"Original RC: {model.params_['RC']:.2f}")
        >>> print(f"Counterfactual RC: {cf.params['RC']:.2f}")
        >>> print(f"P(replace|state=50) changes from "
        ...       f"{model.predict_proba(np.array([50]))[0,1]:.3f} to "
        ...       f"{cf.policy[50,1]:.3f}")
        >>>
        >>> # Multiple parameter changes
        >>> cf2 = model.counterfactual(RC=15.0, theta_c=0.05)
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Check for invalid parameter names
        valid_params = set(self.params_.keys())
        for param_name in param_changes:
            if param_name not in valid_params:
                raise ValueError(
                    f"Unknown parameter '{param_name}'. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )

        # Build counterfactual parameter dictionary
        cf_params = self.params_.copy()
        cf_params.update(param_changes)

        # Build parameter vector in correct order
        param_names = self._result.parameter_names
        param_vector = torch.tensor(
            [cf_params[name] for name in param_names],
            dtype=torch.float32,
        )

        # Compute utility matrix with new parameters
        utility_matrix = self._utility_fn.compute(param_vector)

        # Build transition tensor
        transition_tensor = self._build_transition_tensor(self.transitions_)

        # Create Bellman operator
        operator = SoftBellmanOperator(
            problem=self._problem,
            transitions=transition_tensor,
        )

        # Solve for new value function and policy
        result = value_iteration(operator, utility_matrix)

        return CounterfactualResult(
            params=cf_params,
            value_function=result.V.numpy(),
            policy=result.policy.numpy(),
        )

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"NFXP(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, fitted=True)"
            )
        return (
            f"NFXP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted=False)"
        )
