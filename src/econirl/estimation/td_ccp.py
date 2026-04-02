"""TD-CCP Neural Estimator for dynamic discrete choice models.

This module implements a Temporal Difference CCP estimator that uses
neural networks to approximate the expected value function components.
The approach combines CCP estimation with approximate value iteration (AVI)
using semi-gradient TD learning.

Algorithm:
1. Estimate CCPs from data using frequency estimator
2. Decompose expected value into per-feature flow components plus entropy
3. Train component neural networks via AVI with semi-gradient TD
4. Use learned EV components for partial MLE of structural parameters

This method is particularly useful for large state spaces where
matrix-inversion-based CCP methods become computationally infeasible.

References:
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Sutton, R.S. and Barto, A.G. (2018). "Reinforcement Learning:
        An Introduction." MIT Press.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


@dataclass
class TDCCPConfig:
    """Configuration for the TD-CCP Neural Estimator.

    Attributes:
        hidden_dim: Number of units in each hidden layer of the MLP.
        num_hidden_layers: Number of hidden layers in the MLP.
        avi_iterations: Number of approximate value iteration rounds.
        epochs_per_avi: Number of SGD epochs per AVI iteration.
        learning_rate: Learning rate for NN training.
        batch_size: Mini-batch size for SGD training.
        ccp_smoothing: Smoothing constant added to CCP frequency counts
            to avoid log(0).
        outer_max_iter: Maximum iterations for the L-BFGS-B optimizer
            in the partial MLE step.
        outer_tol: Gradient tolerance for L-BFGS-B convergence.
        compute_se: Whether to compute standard errors via numerical Hessian.
        verbose: Whether to print progress messages.
    """

    hidden_dim: int = 64
    num_hidden_layers: int = 2
    avi_iterations: int = 20
    epochs_per_avi: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 8192
    ccp_smoothing: float = 0.01
    outer_max_iter: int = 200
    outer_tol: float = 1e-6
    n_policy_iterations: int = 3
    policy_iteration_tol: float = 1e-4
    compute_se: bool = True
    verbose: bool = False


class _EVComponentNetwork(eqx.Module):
    """MLP for approximating a single EV component function.

    Maps normalized state features to a scalar value representing
    one component of the expected value decomposition.
    """

    mlp: eqx.nn.MLP

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for a single input vector.

        Args:
            x: Input features of shape (input_dim,).

        Returns:
            Scalar prediction of shape ().
        """
        return self.mlp(x).squeeze(-1)


class TDCCPEstimator(BaseEstimator):
    """TD-CCP Neural Estimator for dynamic discrete choice models.

    Combines CCP estimation with neural network-based approximate value
    iteration. The expected value function is decomposed into per-feature
    components, each learned by a separate MLP using semi-gradient TD.

    The partial MLE step then optimizes structural parameters using
    the learned EV components, avoiding the costly inner fixed-point
    loop required by NFXP.

    Example:
        >>> config = TDCCPConfig(avi_iterations=20, epochs_per_avi=30)
        >>> estimator = TDCCPEstimator(config=config)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        config: TDCCPConfig | None = None,
        se_method: SEMethod = "asymptotic",
        seed: int = 0,
        **kwargs,
    ):
        """Initialize the TD-CCP estimator.

        Args:
            config: Configuration dataclass. Uses defaults if None.
            se_method: Method for computing standard errors.
            seed: Random seed for JAX PRNG key generation.
            **kwargs: Additional keyword arguments (ignored for compatibility).
        """
        if config is None:
            config = TDCCPConfig()
        self._config = config
        self._seed = seed

        super().__init__(
            se_method=se_method,
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )

    @property
    def name(self) -> str:
        """Human-readable name of the estimation method."""
        return "TD-CCP Neural"

    @property
    def config(self) -> TDCCPConfig:
        """Return the configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Step 1: CCP estimation
    # ------------------------------------------------------------------

    def _estimate_ccps(
        self,
        panel: Panel,
        num_states: int,
        num_actions: int,
    ) -> jnp.ndarray:
        """Estimate CCPs from data using frequency estimator with smoothing.

        P_hat(a|s) = (N(s,a) + smoothing) / (N(s) + num_actions * smoothing)

        Args:
            panel: Panel data with observed choices.
            num_states: Number of discrete states.
            num_actions: Number of discrete actions.

        Returns:
            CCP matrix of shape (num_states, num_actions).
        """
        smoothing = self._config.ccp_smoothing
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        counts = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
        counts = counts.at[all_states, all_actions].add(1.0)

        state_counts = counts.sum(axis=1, keepdims=True)
        ccps = (counts + smoothing) / (state_counts + num_actions * smoothing)
        return ccps

    # ------------------------------------------------------------------
    # Step 2: Flow decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_flow_components(
        ccps: jnp.ndarray,
        feature_matrix: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-feature flow functions and entropy flow.

        For each feature k:
            flow_k[s] = sum_a ccps[s,a] * feature_matrix[s,a,k]

        Entropy:
            flow_H[s] = -sum_a ccps[s,a] * log(ccps[s,a])

        Args:
            ccps: CCP matrix of shape (num_states, num_actions).
            feature_matrix: Feature tensor of shape (num_states, num_actions, num_features).

        Returns:
            Tuple of (flow_features, flow_entropy) where:
            - flow_features: shape (num_states, num_features)
            - flow_entropy: shape (num_states,)
        """
        # flow_k[s] = sum_a ccps[s,a] * features[s,a,k]
        flow_features = jnp.einsum("sa,sak->sk", ccps, feature_matrix)

        # Entropy: -sum_a p(a|s) log p(a|s)
        safe_ccps = jnp.clip(ccps, a_min=1e-10)
        flow_entropy = -(ccps * jnp.log(safe_ccps)).sum(axis=1)

        return flow_features, flow_entropy

    # ------------------------------------------------------------------
    # Step 3: Train component networks via AVI with semi-gradient TD
    # ------------------------------------------------------------------

    def _build_state_features(self, states: jnp.ndarray, problem: DDCProblem) -> jnp.ndarray:
        """Create features for NN input from state indices.

        Uses the problem's state_encoder if available, otherwise
        normalizes state index to [0, 1].

        Args:
            states: Array of integer state indices.
            problem: Problem specification with optional state_encoder.

        Returns:
            Feature array of shape (len(states), state_dim).
        """
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        denom = max(problem.num_states - 1, 1)
        return (states.astype(jnp.float32) / denom)[:, None]

    def _train_component_network(
        self,
        flow: jnp.ndarray,
        states: jnp.ndarray,
        next_states: jnp.ndarray,
        problem: DDCProblem,
        gamma: float,
        key: jax.Array,
    ) -> tuple[_EVComponentNetwork, list[float]]:
        """Train a single EV component network via AVI with semi-gradient TD.

        For each AVI round:
            1. Compute targets: Y = flow[s_t] + gamma * net(features[s_{t+1}])
            2. Train net to minimize MSE between net(features[s_t]) and Y
               for epochs_per_avi epochs.

        Args:
            flow: Flow values of shape (num_states,) for this component.
            states: All observed s_t indices (flattened from panel).
            next_states: All observed s_{t+1} indices.
            problem: Problem specification.
            gamma: Discount factor.
            key: JAX PRNG key for network initialization and shuffling.

        Returns:
            Tuple of (trained network, list of per-epoch losses).
        """
        cfg = self._config
        input_dim = problem.state_dim or 1

        key, init_key = jax.random.split(key)
        net = _EVComponentNetwork(input_dim, cfg.hidden_dim, cfg.num_hidden_layers, key=init_key)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(cfg.learning_rate),
        )
        opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

        # Precompute state features for the whole dataset
        feat_s = self._build_state_features(states, problem)
        feat_sp = self._build_state_features(next_states, problem)
        flow_s = flow[states]  # flow values at s_t

        n_samples = len(states)
        losses: list[float] = []

        # Define the JIT-compiled training step. The optimizer is captured
        # in the closure since it is a static (non-array) pytree.
        @eqx.filter_jit
        def train_step(model, opt_state, batch_feat, batch_targets):
            def loss_fn(model):
                preds = jax.vmap(model)(batch_feat)
                return jnp.mean((preds - batch_targets) ** 2)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        from tqdm import tqdm
        pbar = tqdm(
            range(cfg.avi_iterations),
            desc="TD-CCP AVI",
            disable=not self._verbose,
            leave=False,
        )
        for avi_iter in pbar:
            # Compute TD targets with current network (frozen for this AVI round)
            v_next = jax.lax.stop_gradient(jax.vmap(net)(feat_sp))
            targets = flow_s + gamma * v_next

            # Train for epochs_per_avi epochs on these targets
            for epoch in range(cfg.epochs_per_avi):
                # Shuffle indices for mini-batches
                key, perm_key = jax.random.split(key)
                perm = jax.random.permutation(perm_key, n_samples)

                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_samples, cfg.batch_size):
                    end = min(start + cfg.batch_size, n_samples)
                    idx = perm[start:end]

                    net, opt_state, loss = train_step(
                        net, opt_state, feat_s[idx], targets[idx]
                    )

                    epoch_loss += float(loss)
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return net, losses

    def _train_all_components(
        self,
        flow_features: jnp.ndarray,
        flow_entropy: jnp.ndarray,
        panel: Panel,
        problem: DDCProblem,
        gamma: float,
        key: jax.Array,
    ) -> tuple[list[_EVComponentNetwork], _EVComponentNetwork, dict[str, list[float]]]:
        """Train all EV component networks (one per feature + entropy).

        Args:
            flow_features: Per-feature flow values, shape (num_states, num_features).
            flow_entropy: Entropy flow values, shape (num_states,).
            panel: Panel data (for transition samples).
            problem: Problem specification.
            gamma: Discount factor.
            key: JAX PRNG key.

        Returns:
            Tuple of (feature_nets, entropy_net, loss_histories).
        """
        states = panel.get_all_states()
        next_states = panel.get_all_next_states()
        num_features = flow_features.shape[1]

        feature_nets: list[_EVComponentNetwork] = []
        loss_histories: dict[str, list[float]] = {}

        # Train one network per feature component
        for k in range(num_features):
            self._log(f"Training EV component network for feature {k}")
            key, comp_key = jax.random.split(key)
            net, losses = self._train_component_network(
                flow=flow_features[:, k],
                states=states,
                next_states=next_states,
                problem=problem,
                gamma=gamma,
                key=comp_key,
            )
            feature_nets.append(net)
            loss_histories[f"feature_{k}"] = losses

        # Train entropy network
        self._log("Training EV component network for entropy")
        key, entropy_key = jax.random.split(key)
        entropy_net, entropy_losses = self._train_component_network(
            flow=flow_entropy,
            states=states,
            next_states=next_states,
            problem=problem,
            gamma=gamma,
            key=entropy_key,
        )
        loss_histories["entropy"] = entropy_losses

        return feature_nets, entropy_net, loss_histories

    # ------------------------------------------------------------------
    # Step 4: Partial MLE using learned EV components
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_ev_components(
        feature_nets: list[_EVComponentNetwork],
        entropy_net: _EVComponentNetwork,
        state_features: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate all trained component networks on a set of state features.

        Args:
            feature_nets: List of trained networks, one per utility feature.
            entropy_net: Trained entropy network.
            state_features: NN input features, shape (num_states, input_dim).

        Returns:
            Tuple of (ev_features, ev_entropy):
            - ev_features: shape (num_states, num_features)
            - ev_entropy: shape (num_states,)
        """
        ev_list = [jax.vmap(net)(state_features) for net in feature_nets]
        ev_features = jnp.stack(ev_list, axis=1)
        ev_entropy = jax.vmap(entropy_net)(state_features)
        return ev_features, ev_entropy

    def _partial_mle(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        ev_features: jnp.ndarray,
        ev_entropy: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, float, int, int, str]:
        """Optimize structural parameters via partial MLE.

        Using the learned EV components, construct choice-specific values
        and maximize the conditional choice log-likelihood with L-BFGS-B.

        Args:
            panel: Panel data.
            utility: Utility function specification.
            problem: Problem specification.
            transitions: Transition matrices, shape (num_actions, num_states, num_states).
            ev_features: Learned EV for each feature, shape (num_states, num_features).
            ev_entropy: Learned entropy EV, shape (num_states,).
            initial_params: Starting parameter values.

        Returns:
            Tuple of (parameters, log_likelihood, num_iterations, num_feval, message).
        """
        gamma = problem.discount_factor
        sigma = problem.scale_parameter
        num_states = problem.num_states
        num_actions = problem.num_actions
        feature_matrix = utility.feature_matrix  # (S, A, K)

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        def _compute_choice_values(params: jnp.ndarray) -> jnp.ndarray:
            """Compute choice-specific values v(s, a) given parameters.

            v(s, a) = u(s, a; theta)
                      + gamma * sum_k theta_k * (P_a @ ev_k)
                      + gamma * (P_a @ ev_H)

            where P_a is the transition matrix for action a.
            """
            # Flow utility: u(s, a) = theta . phi(s, a)
            flow_u = jnp.einsum("sak,k->sa", feature_matrix, params)

            # Continuation values per action
            # For each action a: E[ev_k | s, a] = transitions[a] @ ev_features[:, k]
            continuation = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
            for a in range(num_actions):
                # Feature part: sum_k theta_k * (P_a @ ev_k)
                ev_next_features = transitions[a] @ ev_features  # (S, K)
                feat_contribution = jnp.einsum("sk,k->s", ev_next_features, params)

                # Entropy part: P_a @ ev_H
                ev_next_entropy = transitions[a] @ ev_entropy  # (S,)

                continuation = continuation.at[:, a].set(
                    gamma * (feat_contribution + ev_next_entropy)
                )

            return flow_u + continuation

        def _log_likelihood(params: jnp.ndarray) -> float:
            """Compute conditional choice log-likelihood."""
            v = _compute_choice_values(params)
            log_probs = jax.nn.log_softmax(v / sigma, axis=1)
            ll = float(log_probs[all_states, all_actions].sum())
            return ll

        # Store the ll function for Hessian computation later
        self._log_likelihood_fn = _log_likelihood
        self._compute_choice_values_fn = _compute_choice_values

        # Scipy optimizer interface
        def objective(params_np):
            params = jnp.array(params_np, dtype=jnp.float32)
            return -_log_likelihood(params)

        def gradient(params_np):
            eps = 1e-5
            n = len(params_np)
            grad = np.zeros(n)
            for i in range(n):
                p_plus = params_np.copy()
                p_minus = params_np.copy()
                p_plus[i] += eps
                p_minus[i] -= eps
                grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * eps)
            return grad

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        lower, upper = utility.get_parameter_bounds()
        bounds = list(zip(np.asarray(lower), np.asarray(upper)))

        result = optimize.minimize(
            objective,
            np.asarray(initial_params),
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={
                "maxiter": self._config.outer_max_iter,
                "gtol": self._config.outer_tol,
            },
        )

        params_opt = jnp.array(result.x, dtype=jnp.float32)
        ll_opt = -result.fun

        return params_opt, ll_opt, result.nit, result.nfev, result.message

    # ------------------------------------------------------------------
    # Main estimation routine
    # ------------------------------------------------------------------

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run TD-CCP with NPL-style policy iteration.

        After the initial CCP-based estimation, recomputes the policy
        from estimated parameters, updates flow components under the
        new policy, retrains EV networks, and re-optimizes. This
        eliminates the scale mismatch from using a fixed CCP.
        """
        start_time = time.time()
        num_states = problem.num_states
        num_actions = problem.num_actions
        gamma = problem.discount_factor
        sigma = problem.scale_parameter
        feature_matrix = utility.feature_matrix

        key = jax.random.PRNGKey(self._seed)

        # Step 1: Initial CCPs from data
        self._log("Step 1: Estimating CCPs from data")
        ccps = self._estimate_ccps(panel, num_states, num_actions)

        total_nit = 0
        total_nfev = 0
        all_loss_histories: dict[str, list[float]] = {}
        params_opt = initial_params

        for pi_iter in range(self._config.n_policy_iterations):
            self._log(f"Policy iteration {pi_iter + 1}/{self._config.n_policy_iterations}")

            # Step 2: Decompose flows under current policy (CCPs)
            flow_features, flow_entropy = self._compute_flow_components(ccps, feature_matrix)

            # Step 3: Train component networks
            self._log("  Training EV component networks via AVI")
            key, train_key = jax.random.split(key)
            feature_nets, entropy_net, loss_histories = self._train_all_components(
                flow_features=flow_features,
                flow_entropy=flow_entropy,
                panel=panel,
                problem=problem,
                gamma=gamma,
                key=train_key,
            )
            for k, v in loss_histories.items():
                all_loss_histories[f"iter{pi_iter}_{k}"] = v

            # Extract learned EV components for all states
            all_state_features = self._build_state_features(
                jnp.arange(num_states), problem
            )
            ev_features, ev_entropy = self._evaluate_ev_components(
                feature_nets, entropy_net, all_state_features
            )

            # Step 4: Partial MLE
            self._log("  Running partial MLE")
            params_opt, ll_opt, n_iter, n_feval, opt_msg = self._partial_mle(
                panel=panel,
                utility=utility,
                problem=problem,
                transitions=transitions,
                ev_features=ev_features,
                ev_entropy=ev_entropy,
                initial_params=params_opt,
            )
            total_nit += n_iter
            total_nfev += n_feval
            self._log(f"  Params: {np.asarray(params_opt)}, LL: {ll_opt:.4f}")

            # Compute exact policy via value iteration with estimated reward
            reward_matrix = utility.compute(params_opt)
            operator = SoftBellmanOperator(problem, transitions)
            vi_result = value_iteration(
                operator, reward_matrix, tol=1e-8, max_iter=5000,
            )
            new_policy = vi_result.policy

            # Check convergence
            policy_change = float(jnp.abs(new_policy - ccps).max())
            self._log(f"  Policy change: {policy_change:.6f}")

            if policy_change < self._config.policy_iteration_tol:
                self._log(f"  Policy converged at iteration {pi_iter + 1}")
                ccps = new_policy
                break

            # Update CCPs for next iteration
            ccps = new_policy

        # Final policy and value via exact value iteration
        reward_matrix = utility.compute(params_opt)
        operator = SoftBellmanOperator(problem, transitions)
        vi_result = value_iteration(operator, reward_matrix, tol=1e-8, max_iter=5000)
        policy = vi_result.policy
        V = vi_result.V

        # Hessian for standard errors
        hessian = None
        if self._config.compute_se:
            self._log("Computing numerical Hessian")

            def ll_fn(params):
                return jnp.array(self._log_likelihood_fn(params))

            hessian = compute_numerical_hessian(params_opt, ll_fn)

        optimization_time = time.time() - start_time

        num_features = utility.num_parameters
        num_components = num_features + 1
        total_inner = (
            self._config.avi_iterations
            * self._config.epochs_per_avi
            * num_components
            * (pi_iter + 1)
        )

        return EstimationResult(
            parameters=params_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            gradient_contributions=None,
            converged=True,
            num_iterations=total_nit,
            num_function_evals=total_nfev,
            num_inner_iterations=total_inner,
            message=f"TD-CCP ({pi_iter + 1} policy iters): {opt_msg}",
            optimization_time=optimization_time,
            metadata={
                "loss_histories": all_loss_histories,
                "avi_iterations": self._config.avi_iterations,
                "epochs_per_avi": self._config.epochs_per_avi,
                "n_policy_iterations": pi_iter + 1,
                "ev_features": ev_features,
                "ev_entropy": ev_entropy,
            },
        )
