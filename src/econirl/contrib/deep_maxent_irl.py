"""Deep Maximum Entropy Inverse Reinforcement Learning estimator.

Extends MaxEnt IRL by parameterizing the reward function R(s,a) with a
neural network instead of linear features. The feature-matching gradient
is backpropagated through the network, enabling non-linear reward recovery.

Algorithm:
    1. Initialize neural reward network R_theta(s, a)
    2. Iteratively:
       a. Compute reward matrix R(s,a) from network
       b. Solve soft Bellman for V and policy pi
       c. Compute state visitation frequencies under pi
       d. Gradient = d/d_theta [empirical_R - expected_R]
       e. Update theta via Adam optimizer
    3. Return learned reward and induced policy

Reference:
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016).
    "Maximum Entropy Deep Inverse Reinforcement Learning."
    arXiv:1507.04888.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import BaseUtilityFunction, UtilityFunction


class _RewardNetwork(eqx.Module):
    """Neural network that maps (state, action) to reward scalar.

    Uses learned embeddings for discrete states and actions, concatenated
    and passed through an MLP.
    """

    state_embedding: jnp.ndarray
    action_embedding: jnp.ndarray
    layers: list

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        embed_dim: int = 16,
        hidden_dims: list[int] | None = None,
        *,
        key: jr.PRNGKey,
    ):
        if hidden_dims is None:
            hidden_dims = [32, 32]

        key, sk, ak = jr.split(key, 3)
        self.state_embedding = jr.normal(sk, (n_states, embed_dim)) * 0.1
        self.action_embedding = jr.normal(ak, (n_actions, embed_dim)) * 0.1

        # Build MLP layers: alternating Linear and relu (stored as callables)
        layers = []
        input_dim = 2 * embed_dim
        for h in hidden_dims:
            key, subkey = jr.split(key)
            layers.append(eqx.nn.Linear(input_dim, h, key=subkey))
            input_dim = h
        key, subkey = jr.split(key)
        layers.append(eqx.nn.Linear(input_dim, 1, key=subkey))

        self.layers = layers

    def _forward_single(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Compute reward for a single (state, action) pair.

        Args:
            state: Single state index, scalar.
            action: Single action index, scalar.

        Returns:
            Reward, scalar.
        """
        se = self.state_embedding[state]
        ae = self.action_embedding[action]
        x = jnp.concatenate([se, ae], axis=-1)

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)

        return jnp.squeeze(x, axis=-1)

    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute rewards for (state, action) pairs.

        Args:
            states: State indices, shape (batch,).
            actions: Action indices, shape (batch,).

        Returns:
            Rewards, shape (batch,).
        """
        return jax.vmap(self._forward_single)(states, actions)

    def reward_matrix(self, n_states: int, n_actions: int) -> jnp.ndarray:
        """Compute full reward matrix R(s, a).

        Returns:
            Reward matrix, shape (n_states, n_actions).
        """
        s_idx = jnp.broadcast_to(jnp.arange(n_states)[:, None], (n_states, n_actions))
        a_idx = jnp.broadcast_to(jnp.arange(n_actions)[None, :], (n_states, n_actions))
        return self(s_idx.reshape(-1), a_idx.reshape(-1)).reshape(n_states, n_actions)


class DeepMaxEntIRLEstimator(BaseEstimator):
    """Deep Maximum Entropy IRL estimator.

    Parameterizes R(s,a) with a neural network and optimizes via
    feature matching with backpropagation. Returns the learned reward
    matrix and the induced policy.

    Since the reward is non-parametric (neural network), this estimator
    cannot recover interpretable structural parameters, but produces
    high-quality policies.

    Attributes:
        hidden_dims: Hidden layer dimensions for the reward network.
        embed_dim: Embedding dimension for states and actions.
        lr: Learning rate for Adam optimizer.
        max_epochs: Maximum training epochs.
        inner_tol: Convergence tolerance for value iteration.
        inner_max_iter: Maximum value iteration iterations.

    Example:
        >>> estimator = DeepMaxEntIRLEstimator(hidden_dims=[32, 32], lr=1e-3)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        embed_dim: int = 16,
        lr: float = 1e-3,
        max_epochs: int = 300,
        gradient_clip: float = 1.0,
        inner_tol: float = 1e-8,
        inner_max_iter: int = 5000,
        compute_se: bool = False,
        verbose: bool = False,
        seed: int = 0,
    ):
        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=verbose,
        )
        self._hidden_dims = hidden_dims if hidden_dims is not None else [32, 32]
        self._embed_dim = embed_dim
        self._lr = lr
        self._max_epochs = max_epochs
        self._gradient_clip = gradient_clip
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._seed = seed

    @property
    def name(self) -> str:
        return "Deep MaxEnt IRL (Wulfmeier 2016)"

    def _compute_state_visitation(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        panel: Panel,
        horizon: int = 100,
    ) -> jnp.ndarray:
        """Compute discounted state visitation frequencies under policy."""
        n_states = problem.num_states
        beta = problem.discount_factor

        # Initial state distribution from data
        init_states = jnp.array(
            [int(traj.states[0]) for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        init_counts = jnp.zeros(n_states).at[init_states].add(1.0)
        mu = init_counts / jnp.clip(init_counts.sum(), a_min=1)

        visitation = mu
        P_pi = jnp.einsum("sa,ast->st", policy, transitions)

        for t in range(1, horizon):
            mu = mu @ P_pi
            visitation = visitation + (beta ** t) * mu

        return visitation / visitation.sum()

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Train neural reward network via Deep MaxEnt IRL."""
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        operator = SoftBellmanOperator(problem, transitions)

        # Compute empirical state-action visitation
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        idx = (all_states * n_actions + all_actions).astype(jnp.int32)
        sa_counts = jnp.zeros(n_states * n_actions).at[idx].add(1.0)
        sa_counts = sa_counts.reshape(n_states, n_actions)
        empirical_sa = sa_counts / sa_counts.sum()

        # Build reward network
        key = jr.PRNGKey(self._seed)
        reward_net = _RewardNetwork(
            n_states, n_actions,
            embed_dim=self._embed_dim,
            hidden_dims=self._hidden_dims,
            key=key,
        )

        # Optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self._gradient_clip),
            optax.adam(self._lr),
        )
        opt_state = optimizer.init(eqx.filter(reward_net, eqx.is_array))

        best_ll = float("-inf")
        best_policy = None
        best_V = None
        best_reward = None

        self._log(f"Training Deep MaxEnt IRL for {self._max_epochs} epochs")

        for epoch in range(self._max_epochs):
            # Forward: compute reward matrix from network
            reward_matrix = reward_net.reward_matrix(n_states, n_actions)

            # Solve MDP (no gradient through VI)
            solver_result = value_iteration(
                operator, reward_matrix,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
            )
            policy = solver_result.policy

            # Compute expected state-action visitation under policy
            state_vis = self._compute_state_visitation(
                policy, transitions, problem, panel,
            )
            expected_sa = jnp.einsum("s,sa->sa", state_vis, policy)

            # Feature matching gradient: the gradient of the loss
            # sum_{s,a} -(empirical_sa - expected_sa) * R(s,a) with respect
            # to network parameters. The coefficient (empirical_sa - expected_sa)
            # is treated as a constant (stop gradient), so only R(s,a) is
            # differentiated through.
            grad_coeff = jax.lax.stop_gradient(empirical_sa - expected_sa)

            def loss_fn(model):
                rm = model.reward_matrix(n_states, n_actions)
                return -(grad_coeff * rm).sum()

            loss, grads = eqx.filter_value_and_grad(loss_fn)(reward_net)
            updates, opt_state = optimizer.update(grads, opt_state, reward_net)
            reward_net = eqx.apply_updates(reward_net, updates)

            # Track best policy by log-likelihood
            rm = reward_net.reward_matrix(n_states, n_actions)
            sr = value_iteration(
                operator, rm,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
            )
            lp = operator.compute_log_choice_probabilities(rm, sr.V)
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            ll = float(lp[all_states, all_actions].sum())

            if ll > best_ll:
                best_ll = ll
                best_policy = jnp.array(sr.policy)
                best_V = jnp.array(sr.V)
                best_reward = jnp.array(rm)

            if self._verbose and (epoch + 1) % 50 == 0:
                self._log(f"  Epoch {epoch+1}: LL={ll:.2f}, best_LL={best_ll:.2f}")

        elapsed = time.time() - start_time

        # Return reward matrix flattened as "parameters" for downstream compatibility
        return EstimationResult(
            parameters=best_reward.flatten(),
            log_likelihood=best_ll,
            value_function=best_V,
            policy=best_policy,
            hessian=None,
            converged=True,
            num_iterations=self._max_epochs,
            message=f"Deep MaxEnt IRL: {self._max_epochs} epochs",
            optimization_time=elapsed,
            metadata={
                "reward_matrix": best_reward,
            },
        )

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate via Deep MaxEnt IRL.

        Overrides base to handle non-parametric reward (neural network).
        """
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        n_obs = panel.num_observations
        n_params = problem.num_states * problem.num_actions

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=result.log_likelihood,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * result.log_likelihood + 2 * n_params,
            bic=-2 * result.log_likelihood
            + n_params * float(jnp.log(n_obs)),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, result.policy
            ),
        )

        param_names = [
            f"R(s={s},a={a})"
            for s in range(problem.num_states)
            for a in range(problem.num_actions)
        ]

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=param_names,
            standard_errors=jnp.full_like(result.parameters, float("nan")),
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=result.log_likelihood,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=True,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=result.optimization_time,
            metadata=result.metadata,
        )
