"""Guided Cost Learning (GCL) Estimator.

This module implements Guided Cost Learning from Finn et al. (2016),
a trajectory-level IRL method that uses neural network cost functions
and importance sampling to estimate the partition function.

Algorithm:
    1. Sample trajectories from current policy q
    2. Compute importance weights w_j = exp(-c(τ)) / q(τ)
    3. Update cost function via gradient descent:
       ∇L = (1/N) Σ ∇c(τ_demo) - Σ w_j ∇c(τ_sample)
    4. Update policy via soft value iteration

Reference:
    Finn, C., Levine, S., & Abbeel, P. (2016). Guided Cost Learning:
    Deep Inverse Optimal Control via Policy Optimization. ICML.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.base import BaseUtilityFunction, UtilityFunction
from econirl.preferences.neural_cost import NeuralCostFunction


@dataclass
class GCLConfig:
    """Configuration for Guided Cost Learning estimation.

    Attributes
    ----------
    embed_dim : int
        Dimension of state and action embeddings.
    hidden_dims : list[int]
        Hidden layer dimensions for the cost network MLP.
    cost_lr : float
        Learning rate for cost network optimization.
    max_iterations : int
        Maximum number of outer loop iterations.
    n_sample_trajectories : int
        Number of trajectories to sample for importance sampling.
    trajectory_length : int
        Length of sampled trajectories.
    importance_clipping : float
        Maximum importance weight (for stability).
    inner_tol : float
        Tolerance for inner soft value iteration.
    inner_max_iter : int
        Maximum iterations for inner soft value iteration.
    switch_tol : float
        Switch tolerance for hybrid iteration.
    verbose : bool
        Print progress messages.
    """

    # Architecture
    embed_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])

    # Optimization
    cost_lr: float = 1e-3
    max_iterations: int = 200
    convergence_tol: float = 1e-4

    # Sampling
    n_sample_trajectories: int = 100
    trajectory_length: int = 50

    # Importance sampling
    importance_clipping: float = 10.0

    # Inner solver (soft value iteration)
    inner_tol: float = 1e-8
    inner_max_iter: int = 5000
    switch_tol: float = 1e-3

    # Normalization
    normalize_reward: bool = False  # Normalize reward to zero mean, unit std
    normalize_features: bool = False  # Normalize state indices to [0, 1]

    # Verbosity
    verbose: bool = False


class GCLEstimator(BaseEstimator):
    """Guided Cost Learning Estimator.

    Implements trajectory-level inverse reinforcement learning using
    neural network cost functions and importance sampling to estimate
    the partition function.

    The objective is to maximize the likelihood of demonstrations:
        L_IOC(θ) = (1/N) Σ c_θ(τ_demo) + log Z

    where Z is approximated via importance sampling:
        Z ≈ (1/M) Σ exp(-c_θ(τ_j)) / q(τ_j)

    Parameters
    ----------
    config : GCLConfig, optional
        Configuration object with algorithm parameters.
    seed : int
        Random seed for JAX PRNG key generation. Default is 42.
    **kwargs
        Override individual config parameters.

    Attributes
    ----------
    config : GCLConfig
        Configuration object.
    cost_function_ : NeuralCostFunction
        Learned neural network cost function (after fitting).

    Examples
    --------
    >>> from econirl.contrib.gcl import GCLEstimator, GCLConfig
    >>>
    >>> config = GCLConfig(verbose=True, max_iterations=100)
    >>> estimator = GCLEstimator(config=config)
    >>> result = estimator.estimate(panel, utility, problem, transitions)
    >>> print(result.policy)  # Learned policy

    References
    ----------
    Finn, C., Levine, S., & Abbeel, P. (2016). Guided Cost Learning:
    Deep Inverse Optimal Control via Policy Optimization. ICML.
    """

    def __init__(
        self,
        config: GCLConfig | None = None,
        seed: int = 42,
        **kwargs,
    ):
        if config is None:
            config = GCLConfig(**kwargs)
        else:
            # Apply any kwargs as overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=config.verbose,
        )
        self.config = config
        self.cost_function_: NeuralCostFunction | None = None
        self._key = jr.key(seed)

    @property
    def name(self) -> str:
        return "GCL (Finn et al. 2016)"

    def _next_key(self) -> jax.Array:
        """Split and advance the internal PRNG key, returning a subkey."""
        self._key, subkey = jr.split(self._key)
        return subkey

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate cost function parameters from panel data.

        GCL uses a neural network cost function, so we override the base
        estimate() method to create an appropriate EstimationSummary.

        Parameters
        ----------
        panel : Panel
            Panel data with observed choices.
        utility : UtilityFunction
            Utility function specification (used for interface compatibility).
        problem : DDCProblem
            Problem specification.
        transitions : jnp.ndarray
            Transition matrices P(s'|s,a).
        initial_params : jnp.ndarray, optional
            Not used (neural network has its own initialization).

        Returns
        -------
        EstimationSummary
            Results with learned cost function and policy.
        """
        import time

        start_time = time.time()

        # Run optimization
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        # Create parameter names for the cost matrix
        n_states = problem.num_states
        n_actions = problem.num_actions
        param_names = [f"c({s},{a})" for s in range(n_states) for a in range(n_actions)]

        # Standard errors not available for neural network parameters
        standard_errors = jnp.full_like(result.parameters, float("nan"))

        # Goodness of fit
        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * np.log(n_obs),
            prediction_accuracy=self._compute_prediction_accuracy(panel, result.policy),
        )

        total_time = time.time() - start_time

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=param_names,
            standard_errors=standard_errors,
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=ll,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=result.converged,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=total_time,
            metadata=result.metadata,
        )

    def _create_cost_function(
        self,
        n_states: int,
        n_actions: int,
    ) -> NeuralCostFunction:
        """Create the neural network cost function."""
        return NeuralCostFunction(
            n_states=n_states,
            n_actions=n_actions,
            embed_dim=self.config.embed_dim,
            hidden_dims=self.config.hidden_dims,
            key=self._next_key(),
        )

    def _sample_trajectories(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        initial_dist: jnp.ndarray,
        n_trajectories: int,
        trajectory_length: int,
    ) -> List[Trajectory]:
        """Sample trajectories from the current policy.

        Parameters
        ----------
        policy : jnp.ndarray
            Policy probabilities pi(a|s), shape (n_states, n_actions).
        transitions : jnp.ndarray
            Transition probabilities P(s'|s,a), shape (n_actions, n_states, n_states).
        initial_dist : jnp.ndarray
            Initial state distribution, shape (n_states,).
        n_trajectories : int
            Number of trajectories to sample.
        trajectory_length : int
            Length of each trajectory.

        Returns
        -------
        trajectories : list[Trajectory]
            Sampled trajectories.
        """
        trajectories = []

        for _ in range(n_trajectories):
            # Sample initial state
            key = self._next_key()
            state = int(jr.categorical(key, jnp.log(initial_dist + 1e-10)))

            states = []
            actions = []
            next_states = []

            for _ in range(trajectory_length):
                # Sample action from policy
                key = self._next_key()
                action_probs = policy[state]
                action = int(jr.categorical(key, jnp.log(action_probs + 1e-10)))

                # Sample next state from transitions
                key = self._next_key()
                trans_probs = transitions[action, state]
                next_state = int(jr.categorical(key, jnp.log(trans_probs + 1e-10)))

                states.append(state)
                actions.append(action)
                next_states.append(next_state)

                state = next_state

            traj = Trajectory(
                states=jnp.array(states, dtype=jnp.int32),
                actions=jnp.array(actions, dtype=jnp.int32),
                next_states=jnp.array(next_states, dtype=jnp.int32),
            )
            trajectories.append(traj)

        return trajectories

    def _compute_trajectory_cost(
        self,
        cost_fn: NeuralCostFunction,
        trajectory: Trajectory,
    ) -> jnp.ndarray:
        """Compute the total cost of a trajectory.

        Parameters
        ----------
        cost_fn : NeuralCostFunction
            Cost function network.
        trajectory : Trajectory
            Trajectory to evaluate.

        Returns
        -------
        total_cost : jnp.ndarray
            Sum of costs along the trajectory (scalar).
        """
        states = trajectory.states
        actions = trajectory.actions
        costs = cost_fn.forward(states, actions)
        return costs.sum()

    def _compute_trajectory_log_prob(
        self,
        policy: jnp.ndarray,
        trajectory: Trajectory,
    ) -> jnp.ndarray:
        """Compute the log probability of a trajectory under the policy.

        Parameters
        ----------
        policy : jnp.ndarray
            Policy probabilities pi(a|s), shape (n_states, n_actions).
        trajectory : Trajectory
            Trajectory to evaluate.

        Returns
        -------
        log_prob : jnp.ndarray
            Log probability of the trajectory (scalar).
        """
        states = trajectory.states
        actions = trajectory.actions

        # Get action probabilities for each state-action pair
        probs = policy[states, actions]

        # Compute log probability
        log_prob = jnp.log(probs + 1e-10).sum()

        return log_prob

    def _compute_importance_weights(
        self,
        cost_fn: NeuralCostFunction,
        policy: jnp.ndarray,
        trajectories: List[Trajectory],
    ) -> jnp.ndarray:
        """Compute normalized importance weights for sampled trajectories.

        The importance weight for trajectory tau is:
            w(tau) proportional to exp(-c(tau)) / q(tau)

        where q(tau) is the probability under the current policy.

        Uses stop_gradient on cost values since we do not want to
        differentiate through the weight computation.

        Parameters
        ----------
        cost_fn : NeuralCostFunction
            Cost function network.
        policy : jnp.ndarray
            Policy probabilities pi(a|s), shape (n_states, n_actions).
        trajectories : list[Trajectory]
            Sampled trajectories.

        Returns
        -------
        weights : jnp.ndarray
            Normalized importance weights, shape (n_trajectories,).
        """
        log_weights = []

        for traj in trajectories:
            cost = jax.lax.stop_gradient(
                self._compute_trajectory_cost(cost_fn, traj)
            )
            log_prob = self._compute_trajectory_log_prob(policy, traj)

            # log w = -c(tau) - log q(tau)
            log_w = -cost - log_prob
            log_weights.append(log_w)

        log_weights = jnp.stack(log_weights)

        # Normalize via softmax for stability
        weights = jax.nn.softmax(log_weights, axis=0)

        # Clip maximum weight for stability
        max_weight = self.config.importance_clipping / len(trajectories)
        weights = jnp.clip(weights, max=max_weight)

        # Re-normalize after clipping
        weights = weights / weights.sum()

        return weights

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute initial state distribution from demonstration data."""
        init_states = jnp.array(
            [int(traj.states[0]) for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        counts = jnp.zeros(n_states, dtype=jnp.float32)
        counts = counts.at[init_states].add(jnp.ones_like(init_states, dtype=jnp.float32))

        total = counts.sum()
        if total > 0:
            return counts / total
        return jnp.ones(n_states) / n_states

    def _update_policy(
        self,
        cost_fn: NeuralCostFunction,
        operator: SoftBellmanOperator,
    ) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        """Update the policy via soft value iteration on the reward.

        Parameters
        ----------
        cost_fn : NeuralCostFunction
            Current cost function.
        operator : SoftBellmanOperator
            Bellman operator.

        Returns
        -------
        V : jnp.ndarray
            Value function, shape (n_states,).
        policy : jnp.ndarray
            Updated policy, shape (n_states, n_actions).
        converged : bool
            Whether soft VI converged.
        """
        # Reward is negative cost
        reward_matrix = -cost_fn.compute()

        # Normalize reward if configured
        if self.config.normalize_reward:
            mean = reward_matrix.mean()
            std = reward_matrix.std()
            if std > 1e-8:
                reward_matrix = (reward_matrix - mean) / std

        # Run soft value iteration
        result = hybrid_iteration(
            operator,
            reward_matrix,
            tol=self.config.inner_tol,
            max_iter=self.config.inner_max_iter,
            switch_tol=self.config.switch_tol,
        )

        return result.V, result.policy, result.converged

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run GCL optimization.

        Parameters
        ----------
        panel : Panel
            Demonstration data.
        utility : BaseUtilityFunction
            Utility function (used for initialization, actual cost is learned).
        problem : DDCProblem
            Problem specification.
        transitions : jnp.ndarray
            Transition probabilities.
        initial_params : jnp.ndarray, optional
            Not used for GCL (neural network has its own initialization).

        Returns
        -------
        result : EstimationResult
            Estimation result with learned cost and policy.
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions

        # Create cost function and operator
        cost_fn = self._create_cost_function(n_states, n_actions)
        operator = SoftBellmanOperator(problem, transitions)

        # Create optax optimizer for cost function
        optimizer = optax.adam(self.config.cost_lr)
        opt_state = optimizer.init(eqx.filter(cost_fn, eqx.is_array))

        # Initial state distribution from demonstrations
        initial_dist = self._compute_initial_distribution(panel, n_states)

        # Initialize policy (uniform)
        policy = jnp.ones((n_states, n_actions)) / n_actions

        # Extract demonstration trajectories for gradient computation
        demo_trajectories = list(panel.trajectories)

        # Tracking
        converged = False
        prev_cost = float('inf')
        current_cost = 0.0
        iteration = 0

        self._log(f"Starting GCL with {len(demo_trajectories)} demonstration trajectories")
        self._log(f"Sampling {self.config.n_sample_trajectories} trajectories per iteration")

        # Progress bar
        pbar = tqdm(
            range(self.config.max_iterations),
            desc="GCL",
            disable=not self.config.verbose,
            leave=True,
        )

        # Define loss function that takes the model as first argument
        # so eqx.filter_value_and_grad can differentiate through it
        def loss_fn(
            cost_fn: NeuralCostFunction,
            demo_trajectories: list[Trajectory],
            sampled_trajectories: list[Trajectory],
            weights: jnp.ndarray,
        ) -> jnp.ndarray:
            # Compute cost for demonstrations
            demo_cost = jnp.float32(0.0)
            for traj in demo_trajectories:
                costs = cost_fn.forward(traj.states, traj.actions)
                demo_cost = demo_cost + costs.sum()
            demo_cost = demo_cost / len(demo_trajectories)

            # Compute weighted cost for samples (importance sampling)
            sample_cost = jnp.float32(0.0)
            for j, traj in enumerate(sampled_trajectories):
                costs = cost_fn.forward(traj.states, traj.actions)
                sample_cost = sample_cost + weights[j] * costs.sum()

            # Loss = E_demo[c] - E_weighted_sample[c]
            # Makes demos lower cost than samples
            return demo_cost - sample_cost

        for iteration in pbar:
            # Step 1: Sample trajectories from current policy
            sampled_trajectories = self._sample_trajectories(
                policy=policy,
                transitions=transitions,
                initial_dist=initial_dist,
                n_trajectories=self.config.n_sample_trajectories,
                trajectory_length=self.config.trajectory_length,
            )

            # Step 2: Compute importance weights (no gradient through weights)
            weights = self._compute_importance_weights(
                cost_fn, policy, sampled_trajectories
            )

            # Step 3: Compute gradient and update cost function
            loss, grads = eqx.filter_value_and_grad(loss_fn)(
                cost_fn, demo_trajectories, sampled_trajectories, weights,
            )
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(cost_fn, eqx.is_array),
            )
            cost_fn = eqx.apply_updates(cost_fn, updates)

            # Step 4: Update policy via soft value iteration
            V, policy, inner_converged = self._update_policy(cost_fn, operator)

            # Check convergence
            current_cost = float(loss)
            cost_change = abs(current_cost - prev_cost)

            pbar.set_postfix({
                "loss": f"{current_cost:.4f}",
                "cost_change": f"{cost_change:.6f}",
            })

            if cost_change < self.config.convergence_tol:
                converged = True
                break

            prev_cost = current_cost

        pbar.close()

        # Store learned cost function
        self.cost_function_ = cost_fn

        # Get final cost and reward matrices
        cost_matrix = cost_fn.compute()
        reward_matrix = -cost_matrix

        # Compute log-likelihood of demonstrations
        V, policy, _ = self._update_policy(cost_fn, operator)
        log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)

        ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())

        optimization_time = time.time() - start_time

        self._log(f"Optimization complete: LL={ll:.2f}, converged={converged}")

        # Return result (parameters are the cost matrix flattened for compatibility)
        return EstimationResult(
            parameters=cost_matrix.flatten(),
            log_likelihood=ll,
            value_function=V,
            policy=policy,
            hessian=None,
            gradient_contributions=None,
            converged=converged,
            num_iterations=iteration + 1,
            num_function_evals=iteration + 1,
            num_inner_iterations=0,
            message="",
            optimization_time=optimization_time,
            metadata={
                "cost_matrix": np.asarray(cost_matrix).tolist(),
                "reward_matrix": np.asarray(reward_matrix).tolist(),
                "final_loss": current_cost,
            },
        )
