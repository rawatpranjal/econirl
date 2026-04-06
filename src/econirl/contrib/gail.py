"""Generative Adversarial Imitation Learning (GAIL) for tabular MDPs.

This module implements GAIL (Ho & Ermon 2016) adapted for discrete choice models
with tabular state-action spaces. Instead of using neural network policies
and RL algorithms, we use soft value iteration for policy optimization.

Algorithm:
    1. Initialize discriminator D(s,a)
    2. Repeat:
       a) Sample state-action pairs from current policy
       b) Sample state-action pairs from expert demonstrations
       c) Update discriminator to classify expert vs policy
       d) Derive reward from discriminator: R(s,a) = -log(1 - D(s,a))
       e) Update policy via soft value iteration with R

Reference:
    Ho, J., & Ermon, S. (2016). "Generative adversarial imitation learning."
    Advances in Neural Information Processing Systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.estimation.adversarial.discriminator import (
    TabularDiscriminator,
    LinearDiscriminator,
)
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class GAILConfig:
    """Configuration for tabular GAIL.

    Attributes:
        discriminator_type: Type of discriminator ("tabular" or "linear")
        discriminator_lr: Learning rate for discriminator updates
        discriminator_steps: Discriminator updates per round
        generator_solver: Inner solver for policy ("value" or "hybrid")
        generator_tol: Tolerance for value iteration
        generator_max_iter: Max iterations for value iteration
        max_rounds: Maximum training rounds
        batch_size: Batch size for sampling (0 = use all data)
        entropy_coef: Entropy regularization coefficient
        compute_se: Whether to compute standard errors
        se_method: Method for standard errors
        n_bootstrap: Number of bootstrap samples
        convergence_tol: Tolerance for policy convergence
        verbose: Whether to print progress
        seed: Random seed for JAX PRNG
    """

    discriminator_type: Literal["tabular", "linear"] = "tabular"
    discriminator_lr: float = 0.01
    discriminator_steps: int = 5
    generator_solver: Literal["value", "hybrid"] = "hybrid"
    generator_tol: float = 1e-8
    generator_max_iter: int = 5000
    policy_step_size: float = 1.0  # Conservative policy iteration mixing.
    # 1.0 = full VI (original). <1 = mix old/new policy to prevent instability.
    max_rounds: int = 100
    batch_size: int = 0  # 0 means use all
    entropy_coef: float = 0.0
    reward_transform: Literal["softplus", "logit"] = "softplus"  # "logit" uses raw D(s,a)
    convergence_tol: float = 1e-4
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic"] = "bootstrap"
    n_bootstrap: int = 100
    verbose: bool = False
    seed: int = 42


class GAILEstimator(BaseEstimator):
    """Generative Adversarial Imitation Learning for tabular MDPs.

    GAIL learns a reward function and policy by training a discriminator
    to distinguish expert behavior from the current policy. The policy
    is then updated to maximize the learned reward.

    For tabular MDPs, we use:
    - TabularDiscriminator or LinearDiscriminator instead of neural nets
    - Soft value iteration instead of RL algorithms (PPO, SAC)

    Parameters
    ----------
    config : GAILConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Examples
    --------
    >>> from econirl.estimation.adversarial import GAILEstimator, GAILConfig
    >>> config = GAILConfig(max_rounds=100, verbose=True)
    >>> estimator = GAILEstimator(config=config)
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    """

    def __init__(
        self,
        config: GAILConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = GAILConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method=config.se_method if config.compute_se else "asymptotic",
            compute_hessian=False,  # GAIL doesn't use Hessian
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "GAIL (Ho & Ermon 2016)"

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate reward function using GAIL.

        Overrides base class to handle discriminator parameters properly.
        For tabular GAIL, "parameters" are the discriminator logits.
        For linear GAIL, "parameters" are the discriminator weights.

        Args:
            panel: Expert demonstrations
            utility: Utility/reward function specification (used for features in linear mode)
            problem: Problem specification
            transitions: Transition matrices
            initial_params: Initial parameters (optional)

        Returns:
            EstimationSummary with learned parameters and policy
        """
        import time as time_module

        start_time = time_module.time()

        # Run optimization
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        # Generate parameter names for discriminator
        if self.config.discriminator_type == "linear":
            # Linear discriminator has weights matching utility features
            param_names = utility.parameter_names
        else:
            # Tabular discriminator: one parameter per (state, action) pair
            param_names = [
                f"D({s},{a})"
                for s in range(problem.num_states)
                for a in range(problem.num_actions)
            ]

        # Create standard errors (NaN for adversarial methods)
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
            bic=-2 * ll + n_params * float(jnp.log(jnp.array(n_obs))),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, result.policy
            ),
        )

        total_time = time_module.time() - start_time

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

    def _sample_from_panel(
        self,
        panel: Panel,
        batch_size: int | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample state-action pairs from expert demonstrations.

        Args:
            panel: Panel data with expert trajectories
            batch_size: Number of samples (None = all)
            key: JAX PRNG key (required when batch_size is used)

        Returns:
            Tuple of (states, actions) arrays
        """
        states = panel.get_all_states()
        actions = panel.get_all_actions()

        if batch_size is not None and batch_size > 0 and batch_size < len(states):
            indices = jr.choice(key, len(states), shape=(batch_size,), replace=False)
            return states[indices], actions[indices]

        return states, actions

    def _sample_from_policy(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        n_samples: int,
        initial_dist: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample state-action pairs from current policy.

        Simulates trajectories under the policy and collects (s, a) pairs.

        Args:
            policy: Current policy pi(a|s), shape (n_states, n_actions)
            transitions: Transition matrices P(s'|s,a)
            n_samples: Number of samples to collect
            initial_dist: Initial state distribution
            key: JAX PRNG key

        Returns:
            Tuple of (states, actions) arrays
        """
        n_states, n_actions = policy.shape
        states = []
        actions = []

        # Start from initial distribution
        key, subkey = jr.split(key)
        state = int(jr.categorical(subkey, jnp.log(initial_dist)))

        for _ in range(n_samples):
            # Sample action from policy
            key, subkey = jr.split(key)
            action = int(jr.categorical(subkey, jnp.log(policy[state])))
            states.append(state)
            actions.append(action)

            # Sample next state from transition
            key, subkey = jr.split(key)
            next_state_dist = transitions[action, state, :]
            state = int(jr.categorical(subkey, jnp.log(next_state_dist)))

        return jnp.array(states, dtype=jnp.int32), jnp.array(
            actions, dtype=jnp.int32
        )

    def _compute_policy(
        self,
        reward_matrix: jnp.ndarray,
        operator: SoftBellmanOperator,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute optimal policy given reward matrix.

        Args:
            reward_matrix: Reward R(s,a), shape (n_states, n_actions)
            operator: Soft Bellman operator

        Returns:
            Tuple of (policy, value_function)
        """
        if self.config.generator_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                reward_matrix,
                tol=self.config.generator_tol,
                max_iter=self.config.generator_max_iter,
            )
        else:
            result = value_iteration(
                operator,
                reward_matrix,
                tol=self.config.generator_tol,
                max_iter=self.config.generator_max_iter,
            )
        return result.policy, result.V

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute initial state distribution from data."""
        init_states = jnp.array(
            [int(traj.states[0]) for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        counts = jnp.zeros(n_states, dtype=jnp.float32).at[init_states].add(
            jnp.ones_like(init_states, dtype=jnp.float32)
        )

        total = counts.sum()
        if total > 0:
            return counts / total
        return jnp.ones(n_states) / n_states

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run GAIL optimization.

        Args:
            panel: Expert demonstrations
            utility: Reward function (used for feature extraction in linear discriminator)
            problem: Problem specification
            transitions: Transition matrices
            initial_params: Initial reward weights (optional)

        Returns:
            EstimationResult with learned policy and pseudo-parameters
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        operator = SoftBellmanOperator(problem, transitions)

        # Initialize JAX PRNG key
        key = jr.PRNGKey(self.config.seed)

        # Initialize discriminator
        if self.config.discriminator_type == "linear":
            if isinstance(utility, ActionDependentReward):
                feature_matrix = utility.feature_matrix
            elif isinstance(utility, LinearReward):
                # Broadcast state features to action-dependent
                sf = utility.state_features  # (n_states, n_features)
                feature_matrix = jnp.broadcast_to(
                    sf[:, None, :], (sf.shape[0], n_actions, sf.shape[-1])
                )
            else:
                raise TypeError(f"Unsupported utility type: {type(utility)}")
            discriminator = LinearDiscriminator(feature_matrix)
        else:
            discriminator = TabularDiscriminator(n_states, n_actions, init="zeros")

        # Initial state distribution
        initial_dist = self._compute_initial_distribution(panel, n_states)

        # Sample expert data once
        expert_states, expert_actions = self._sample_from_panel(panel)
        n_expert = len(expert_states)

        # Initialize policy (uniform)
        policy = jnp.ones((n_states, n_actions)) / n_actions
        V = jnp.zeros(n_states)

        # Track best policy by log-likelihood (adversarial training oscillates)
        best_ll = float('-inf')
        best_policy = jnp.array(policy)
        best_V = jnp.array(V)
        best_reward = None

        # Training metrics
        disc_losses = []
        policy_changes = []
        converged = False
        round_idx = 0

        pbar = tqdm(
            range(self.config.max_rounds),
            desc="GAIL",
            disable=not self.config.verbose,
        )

        for round_idx in pbar:
            old_policy = jnp.array(policy)

            # Sample from current policy
            key, subkey = jr.split(key)
            policy_states, policy_actions = self._sample_from_policy(
                policy, transitions, n_expert, initial_dist, subkey
            )

            # Update discriminator
            batch_size = (
                self.config.batch_size if self.config.batch_size > 0 else n_expert
            )
            disc_loss = 0.0

            for _ in range(self.config.discriminator_steps):
                # Sample batches
                if batch_size < n_expert:
                    key, subkey1, subkey2 = jr.split(key, 3)
                    expert_idx = jr.choice(
                        subkey1, n_expert, shape=(batch_size,), replace=False
                    )
                    policy_idx = jr.choice(
                        subkey2, n_expert, shape=(batch_size,), replace=False
                    )
                    e_states, e_actions = (
                        expert_states[expert_idx],
                        expert_actions[expert_idx],
                    )
                    p_states, p_actions = (
                        policy_states[policy_idx],
                        policy_actions[policy_idx],
                    )
                else:
                    e_states, e_actions = expert_states, expert_actions
                    p_states, p_actions = policy_states, policy_actions

                disc_loss = discriminator.update(
                    e_states,
                    e_actions,
                    p_states,
                    p_actions,
                    learning_rate=self.config.discriminator_lr,
                )
            disc_losses.append(disc_loss)

            # Derive reward from discriminator
            if self.config.reward_transform == "logit":
                reward_matrix = discriminator.get_reward_matrix(reward_type="airl")
            else:
                reward_matrix = discriminator.get_reward_matrix(reward_type="gail")

            # Add entropy bonus if specified
            if self.config.entropy_coef > 0:
                # Entropy bonus: encourage exploration
                reward_matrix = reward_matrix + self.config.entropy_coef * jnp.log(
                    policy + 1e-10
                )

            # Update policy via soft value iteration
            new_policy, V = self._compute_policy(reward_matrix, operator)

            # Conservative policy iteration: mix old and new
            alpha = self.config.policy_step_size
            if alpha < 1.0:
                policy = (1 - alpha) * old_policy + alpha * new_policy
                policy = policy / policy.sum(axis=1, keepdims=True)
            else:
                policy = new_policy

            # Track best policy by log-likelihood on expert data
            log_probs_iter = operator.compute_log_choice_probabilities(reward_matrix, V)
            ll_iter = float(log_probs_iter[panel.get_all_states(), panel.get_all_actions()].sum())
            if ll_iter > best_ll:
                best_ll = ll_iter
                best_policy = jnp.array(policy)
                best_V = jnp.array(V)
                best_reward = jnp.array(reward_matrix)

            # Check convergence
            policy_change = float(jnp.abs(policy - old_policy).max())
            policy_changes.append(policy_change)

            pbar.set_postfix({
                "d_loss": f"{disc_loss:.4f}",
                "d_pol": f"{policy_change:.4f}",
                "P(R|hi)": f"{float(policy[-10:, 1].mean()):.3f}",
                "best_ll": f"{best_ll:.1f}",
            })

            if policy_change < self.config.convergence_tol:
                converged = True
                break

        pbar.close()

        # Use best policy found during training (adversarial training oscillates)
        policy = best_policy
        V = best_V
        final_reward = best_reward if best_reward is not None else (
            discriminator.get_reward_matrix(reward_type="airl" if self.config.reward_transform == "logit" else "gail")
        )

        # Compute pseudo log-likelihood
        log_probs = operator.compute_log_choice_probabilities(final_reward, V)
        ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())

        # Extract "parameters" (discriminator weights or reward values)
        if isinstance(discriminator, LinearDiscriminator):
            parameters = jnp.array(discriminator.weights)
        else:
            # For tabular discriminator, flatten the logits as pseudo-parameters
            parameters = discriminator.logits.flatten()

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=parameters,
            log_likelihood=ll,
            value_function=V,
            policy=policy,
            hessian=None,
            converged=converged,
            num_iterations=round_idx + 1,
            num_function_evals=round_idx + 1,
            message="Converged" if converged else "Max rounds reached",
            optimization_time=optimization_time,
            metadata={
                "discriminator_type": self.config.discriminator_type,
                "final_disc_loss": disc_losses[-1] if disc_losses else None,
                "disc_losses": disc_losses,
                "policy_changes": policy_changes,
                "reward_matrix": final_reward.tolist(),
            },
        )
