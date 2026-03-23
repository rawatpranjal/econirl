"""Sklearn-style Guided Cost Learning estimator.

Guided Cost Learning (Finn et al. 2016) with sklearn-style API.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.gcl import GCLEstimator, GCLConfig
from econirl.preferences.neural_cost import NeuralCostFunction
from econirl.transitions import TransitionEstimator


class GCL:
    """Sklearn-style Guided Cost Learning estimator.

    Guided Cost Learning (Finn et al. 2016) recovers a neural network
    cost function from demonstrated behavior using importance sampling.

    Unlike linear IRL methods, GCL learns a flexible cost function
    parameterized by a neural network, allowing it to capture complex
    cost structures.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states.
    n_actions : int, default=2
        Number of discrete actions.
    discount : float, default=0.99
        Time discount factor (beta).
    embed_dim : int, default=32
        Dimension of state and action embeddings.
    hidden_dims : list[int], default=[64, 64]
        Hidden layer dimensions for the cost network MLP.
    cost_lr : float, default=1e-3
        Learning rate for cost network optimization.
    max_iterations : int, default=200
        Maximum number of outer loop iterations.
    n_sample_trajectories : int, default=100
        Number of trajectories to sample per iteration.
    trajectory_length : int, default=50
        Length of sampled trajectories.
    importance_clipping : float, default=10.0
        Maximum importance weight (for stability).
    verbose : bool, default=False
        Print progress messages.

    Attributes
    ----------
    cost_matrix_ : numpy.ndarray
        Learned cost matrix c(s, a), shape (n_states, n_actions).
    reward_matrix_ : numpy.ndarray
        Reward matrix R(s, a) = -c(s, a), shape (n_states, n_actions).
    policy_ : numpy.ndarray
        Learned policy π(a|s), shape (n_states, n_actions).
    value_function_ : numpy.ndarray
        Value function V(s) for each state.
    log_likelihood_ : float
        Log-likelihood of the data under learned model.
    converged_ : bool
        Whether optimization converged.
    cost_function_ : NeuralCostFunction
        The learned neural network cost function.

    Examples
    --------
    >>> from econirl.estimators import GCL
    >>> from econirl.datasets import load_rust_bus
    >>>
    >>> df = load_rust_bus()
    >>>
    >>> model = GCL(
    ...     n_states=90,
    ...     n_actions=2,
    ...     discount=0.99,
    ...     embed_dim=32,
    ...     hidden_dims=[64, 64],
    ...     verbose=True,
    ... )
    >>> model.fit(df, state='mileage_bin', action='replaced', id='bus_id')
    >>> print(model.summary())
    >>> print('Cost matrix shape:', model.cost_matrix_.shape)

    References
    ----------
    Finn, C., Levine, S., & Abbeel, P. (2016). Guided Cost Learning:
    Deep Inverse Optimal Control via Policy Optimization. ICML.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        embed_dim: int = 32,
        hidden_dims: list[int] | None = None,
        cost_lr: float = 1e-3,
        max_iterations: int = 200,
        n_sample_trajectories: int = 100,
        trajectory_length: int = 50,
        importance_clipping: float = 10.0,
        normalize_reward: bool = False,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 64]
        self.cost_lr = cost_lr
        self.max_iterations = max_iterations
        self.n_sample_trajectories = n_sample_trajectories
        self.trajectory_length = trajectory_length
        self.importance_clipping = importance_clipping
        self.normalize_reward = normalize_reward
        self.verbose = verbose

        # Fitted attributes
        self.cost_matrix_: np.ndarray | None = None
        self.reward_matrix_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_function_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.converged_: bool | None = None
        self.cost_function_: NeuralCostFunction | None = None

        # Internal
        self._result = None
        self._panel = None
        self._problem = None
        self._estimator = None

    def fit(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        transitions: np.ndarray | None = None,
    ) -> "GCL":
        """Fit the GCL estimator.

        Parameters
        ----------
        data : pandas.DataFrame
            Panel data with demonstrations.
        state : str
            Column name for state variable.
        action : str
            Column name for action variable.
        id : str
            Column name for individual/trajectory identifier.
        transitions : numpy.ndarray, optional
            Pre-estimated transition matrix (n_states, n_states) for
            the "keep" action. If None, estimated from data.

        Returns
        -------
        self : GCL
            Fitted estimator.
        """
        # Convert to Panel
        self._panel = self._dataframe_to_panel(data, state, action, id)

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

        # Create a dummy utility function (GCL learns its own cost function)
        dummy_cost = NeuralCostFunction(
            n_states=self.n_states,
            n_actions=self.n_actions,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
        )

        # Create estimator with config
        config = GCLConfig(
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            cost_lr=self.cost_lr,
            max_iterations=self.max_iterations,
            n_sample_trajectories=self.n_sample_trajectories,
            trajectory_length=self.trajectory_length,
            importance_clipping=self.importance_clipping,
            normalize_reward=self.normalize_reward,
            verbose=self.verbose,
        )
        self._estimator = GCLEstimator(config=config)

        # Estimate
        self._result = self._estimator.estimate(
            panel=self._panel,
            utility=dummy_cost,
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
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _build_transition_tensor(self, keep_transitions: np.ndarray) -> torch.Tensor:
        """Build transition tensor for both actions."""
        n = self.n_states
        transitions = torch.zeros((self.n_actions, n, n), dtype=torch.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = torch.tensor(keep_transitions, dtype=torch.float32)

        # Action 1 (replace): reset to state 0, then transition
        for s in range(n):
            transitions[1, s, :] = transitions[0, 0, :]

        return transitions

    def _extract_results(self) -> None:
        """Extract results into sklearn-style attributes."""
        if self._result is None:
            return

        # Extract cost and reward matrices
        if self._result.metadata and "cost_matrix" in self._result.metadata:
            self.cost_matrix_ = np.array(self._result.metadata["cost_matrix"])
            self.reward_matrix_ = np.array(self._result.metadata["reward_matrix"])
        else:
            # Fallback: reshape parameters
            self.cost_matrix_ = self._result.parameters.detach().numpy().reshape(
                self.n_states, self.n_actions
            )
            self.reward_matrix_ = -self.cost_matrix_

        # Policy
        if self._result.policy is not None:
            self.policy_ = self._result.policy.detach().numpy()

        # Value function
        if self._result.value_function is not None:
            self.value_function_ = self._result.value_function.detach().numpy()

        # Store the learned cost function
        if self._estimator is not None:
            self.cost_function_ = self._estimator.cost_function_

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

    def get_cost(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get learned costs for state-action pairs.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.
        actions : numpy.ndarray
            Array of action indices.

        Returns
        -------
        costs : numpy.ndarray
            Cost values for each (state, action) pair.
        """
        if self.cost_matrix_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)

        return self.cost_matrix_[states, actions]

    def get_reward(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get learned rewards for state-action pairs.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.
        actions : numpy.ndarray
            Array of action indices.

        Returns
        -------
        rewards : numpy.ndarray
            Reward values (negative costs) for each (state, action) pair.
        """
        if self.reward_matrix_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)

        return self.reward_matrix_[states, actions]

    def summary(self) -> str:
        """Generate formatted summary of results."""
        if self._result is None:
            return "GCL: Not fitted yet. Call fit() first."

        lines = []
        lines.append("=" * 70)
        lines.append("Guided Cost Learning Results".center(70))
        lines.append("=" * 70)
        lines.append(f"{'Method:':<25} GCL (Finn et al. 2016)")
        lines.append(f"{'Discount Factor (β):':<25} {self.discount}")
        lines.append(f"{'No. States:':<25} {self.n_states}")
        lines.append(f"{'No. Actions:':<25} {self.n_actions}")
        lines.append(f"{'Log-Likelihood:':<25} {self.log_likelihood_:,.2f}")
        lines.append(f"{'Converged:':<25} {'Yes' if self.converged_ else 'No'}")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Network Architecture:")
        lines.append("-" * 70)
        lines.append(f"{'Embedding Dimension:':<25} {self.embed_dim}")
        lines.append(f"{'Hidden Layers:':<25} {self.hidden_dims}")
        lines.append(f"{'Learning Rate:':<25} {self.cost_lr}")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Cost Matrix Statistics:")
        lines.append("-" * 70)

        if self.cost_matrix_ is not None:
            cost_flat = self.cost_matrix_.flatten()
            lines.append(f"{'Min cost:':<25} {cost_flat.min():.4f}")
            lines.append(f"{'Max cost:':<25} {cost_flat.max():.4f}")
            lines.append(f"{'Mean cost:':<25} {cost_flat.mean():.4f}")
            lines.append(f"{'Std cost:':<25} {cost_flat.std():.4f}")

            # Cost difference between actions
            if self.n_actions == 2:
                cost_diff = self.cost_matrix_[:, 1] - self.cost_matrix_[:, 0]
                lines.append(f"{'Mean cost(a=1) - cost(a=0):':<25} {cost_diff.mean():.4f}")

        lines.append("-" * 70)
        lines.append("")
        lines.append("Policy Statistics:")
        lines.append("-" * 70)

        if self.policy_ is not None:
            # Fraction choosing each action
            mean_probs = self.policy_.mean(axis=0)
            for a in range(self.n_actions):
                lines.append(f"{'Mean P(a=' + str(a) + '):':<25} {mean_probs[a]:.4f}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = self.cost_matrix_ is not None
        return (
            f"GCL(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, embed_dim={self.embed_dim}, "
            f"hidden_dims={self.hidden_dims}, fitted={fitted})"
        )
