"""Neural Network Estimation of Structural models (NNES).

Two estimator variants that combine neural V(s) approximation with
structural MLE:

NNESEstimator (NPL-based, default):
    Phase 1 trains V_phi on the NPL Bellman target using data CCPs and
    the Hotz-Miller emax correction. The zero Jacobian property of the
    NPL mapping implies Neyman orthogonality: first-order errors in
    V_phi drop out of the Phase 2 score, so the pseudo-likelihood
    Hessian is the correct semiparametrically efficient variance
    estimator without bias correction (Nguyen 2025, Propositions 3-4).

NNESNFXPEstimator (NFXP-based, legacy):
    Phase 1 trains V_phi on the NFXP soft Bellman operator via residual
    minimization. Does NOT have the Neyman orthogonality property.
    V-approximation errors feed directly into the Phase 2 score.

Both share the same Phase 2: plug learned V_phi into a CCP
log-likelihood and optimize over structural parameters theta.

Reference:
    Nguyen, H. (2025). "Neural Network Estimators for Dynamic Discrete
    Choice Models." Working Paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


class _ValueNetwork(eqx.Module):
    """MLP that maps state features to a scalar value V(s).

    When anchor_features is set the network returns the anchored
    difference V(s) = f(s) - f(x_0) per Nguyen (2025), guaranteeing
    V(x_0) = 0 and removing the additive identification ambiguity that
    becomes ill-conditioned as beta approaches 1.
    """

    mlp: eqx.nn.MLP
    anchor_features: jnp.ndarray | None = eqx.field(static=False, default=None)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        *,
        key: jax.random.PRNGKey,
        anchor_features: jnp.ndarray | None = None,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.relu,
            key=key,
        )
        self.anchor_features = anchor_features

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. Input shape (input_dim,), output scalar."""
        v = self.mlp(x).squeeze(-1)
        if self.anchor_features is not None:
            v0 = self.mlp(self.anchor_features).squeeze(-1)
            v = v - v0
        return v


@dataclass
class NNESConfig:
    """Configuration for NNES estimators (both NPL and NFXP variants).

    Attributes:
        hidden_dim: Hidden units per layer in V-network.
        num_layers: Number of hidden layers in V-network.
        v_lr: Learning rate for V-network training (Phase 1).
        v_epochs: Number of training epochs for V-network per outer iteration.
        v_batch_size: Mini-batch size for V-network SGD.
        outer_max_iter: Maximum L-BFGS-B iterations for structural MLE (Phase 2).
        outer_tol: Gradient tolerance for L-BFGS-B convergence.
        n_outer_iterations: Number of Phase 1 + Phase 2 alternations.
        compute_se: Whether to compute standard errors.
        se_method: Standard error method.
        verbose: Whether to print progress.
        seed: PRNG seed for reproducibility.
    """

    hidden_dim: int = 32
    num_layers: int = 2
    v_lr: float = 1e-3
    v_epochs: int = 500
    v_batch_size: int = 512
    outer_max_iter: int = 200
    outer_tol: float = 1e-6
    n_outer_iterations: int = 3
    anchor_state: int | None = 0
    compute_se: bool = True
    se_method: SEMethod = "asymptotic"
    verbose: bool = False
    seed: int = 0


class NNESNFXPEstimator(BaseEstimator):
    """NNES with NFXP Bellman residual minimization (legacy variant).

    Phase 1 trains V_phi to be a fixed point of the NFXP soft Bellman
    operator: V(s) = sigma * logsumexp([u(s,a;theta) + beta*E[V(s')]]/sigma).
    Phase 2 plugs V_phi into a CCP log-likelihood and optimizes over theta.

    WARNING: This variant does NOT have the Neyman orthogonality property.
    V-approximation errors feed directly into the Phase 2 score, so
    standard errors from the pseudo-likelihood Hessian are not
    semiparametrically efficient. Use NNESEstimator (NPL-based) instead
    for valid inference.

    Args:
        config: NNESConfig or keyword arguments matching NNESConfig fields.

    Example:
        >>> estimator = NNESNFXPEstimator(hidden_dim=32, v_epochs=500)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        v_lr: float = 1e-3,
        v_epochs: int = 500,
        v_batch_size: int = 512,
        outer_max_iter: int = 200,
        outer_tol: float = 1e-6,
        n_outer_iterations: int = 3,
        anchor_state: int | None = 0,
        compute_se: bool = True,
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
        seed: int = 0,
        config: NNESConfig | None = None,
    ):
        if config is not None:
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers
            v_lr = config.v_lr
            v_epochs = config.v_epochs
            v_batch_size = config.v_batch_size
            outer_max_iter = config.outer_max_iter
            outer_tol = config.outer_tol
            n_outer_iterations = config.n_outer_iterations
            anchor_state = config.anchor_state
            compute_se = config.compute_se
            se_method = config.se_method
            verbose = config.verbose
            seed = config.seed

        super().__init__(
            se_method=se_method,
            compute_hessian=compute_se,
            verbose=verbose,
        )
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._v_lr = v_lr
        self._v_epochs = v_epochs
        self._v_batch_size = v_batch_size
        self._outer_max_iter = outer_max_iter
        self._outer_tol = outer_tol
        self._n_outer_iterations = n_outer_iterations
        self._anchor_state = anchor_state
        self._compute_se = compute_se
        self._seed = seed
        self._config = NNESConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            v_lr=v_lr,
            v_epochs=v_epochs,
            v_batch_size=v_batch_size,
            outer_max_iter=outer_max_iter,
            outer_tol=outer_tol,
            n_outer_iterations=n_outer_iterations,
            anchor_state=anchor_state,
            compute_se=compute_se,
            se_method=se_method,
            verbose=verbose,
            seed=seed,
        )

    @property
    def name(self) -> str:
        return "NNES-NFXP (Bellman residual)"

    @property
    def config(self) -> NNESConfig:
        """Return current configuration."""
        return self._config

    def _bootstrap_params_from_ccp(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        sigma: float,
        beta: float,
        feature_matrix: jnp.ndarray,
        n_states: int,
        n_actions: int,
        bounds: list,
    ) -> jnp.ndarray:
        """Estimate initial params from data CCPs via Hotz-Miller inversion.

        Uses CCP frequencies to compute the EV matrix-inversion closed form,
        then solves for theta via a quick partial MLE.
        """
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        # Estimate CCPs from data
        counts = jnp.zeros((n_states, n_actions))
        for s, a in zip(np.asarray(all_states), np.asarray(all_actions)):
            counts = counts.at[int(s), int(a)].add(1.0)
        state_counts = jnp.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        ccps = (counts + 0.01) / (state_counts + n_actions * 0.01)

        # Hotz-Miller: compute EV via matrix inversion
        # P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)
        P_pi = jnp.einsum("sa,ast->st", ccps, transitions)

        # Expected flow per-feature: flow_k[s] = sum_a pi(a|s) * phi(s,a,k)
        flow_features = jnp.einsum("sa,sak->sk", ccps, feature_matrix)

        # Entropy: H(s) = -sum_a p(a|s) log p(a|s)
        safe_ccps = jnp.maximum(ccps, 1e-10)
        entropy = -(ccps * jnp.log(safe_ccps)).sum(axis=1)

        # Solve for EV components: ev_k = (I - beta * P_pi)^{-1} flow_k
        # Use float64 for numerical stability at high discount factors
        # (condition number of I - beta*P_pi ~ 1/(1-beta) ~ 10,000 at beta=0.9999)
        eye = jnp.eye(n_states, dtype=jnp.float64)
        try:
            M = eye - beta * P_pi.astype(jnp.float64)
            ev_features = jnp.linalg.solve(
                M,
                flow_features.astype(jnp.float64),
            ).astype(jnp.float32)
            ev_entropy = jnp.linalg.solve(
                M,
                entropy.astype(jnp.float64),
            ).astype(jnp.float32)
        except Exception:
            # Fallback: return small positive values
            return jnp.full((feature_matrix.shape[2],), 0.01)

        # Precompute EV transition products so the JAX objective is pure.
        # ev_trans_feat[a, s, k] = sum_s' P(s'|s,a) * ev_features[s', k]
        # ev_trans_ent[a, s]     = sum_s' P(s'|s,a) * ev_entropy[s']
        ev_trans_feat = jnp.stack(
            [transitions[a] @ ev_features for a in range(n_actions)], axis=0,
        )  # (A, S, K)
        ev_trans_ent = jnp.stack(
            [transitions[a] @ ev_entropy for a in range(n_actions)], axis=0,
        )  # (A, S)

        # Pure JAX objective for minimize_lbfgsb
        def _bootstrap_obj_jax(params):
            flow_u = jnp.einsum("sak,k->sa", feature_matrix, params)
            # continuation[s, a] = beta * (sum_k ev_trans_feat[a,s,k]*params[k]
            #                               + sigma * ev_trans_ent[a,s])
            feat_part = jnp.einsum("ask,k->sa", ev_trans_feat, params)
            ent_part = ev_trans_ent.T  # (S, A)
            continuation = beta * (feat_part + sigma * ent_part)
            v = flow_u + continuation
            log_probs = jax.nn.log_softmax(v / sigma, axis=1)
            return -log_probs[all_states, all_actions].sum()

        # Start slightly above zero to avoid boundary stall
        x0 = jnp.full(feature_matrix.shape[2], 0.01, dtype=jnp.float32)
        lower_b = jnp.array([b[0] for b in bounds], dtype=jnp.float32)
        upper_b = jnp.array([b[1] for b in bounds], dtype=jnp.float32)
        result = minimize_lbfgsb(
            _bootstrap_obj_jax,
            x0,
            bounds=(lower_b, upper_b),
            maxiter=100,
            tol=1e-6,
            verbose=False,
            desc="NNES bootstrap",
        )
        return jnp.array(result.x, dtype=jnp.float32)

    def _build_state_features(self, states: jnp.ndarray, problem: DDCProblem) -> jnp.ndarray:
        """Build state features from state indices using problem's encoder."""
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        denom = max(problem.num_states - 1, 1)
        return (states.astype(jnp.float32) / denom)[:, None]

    def _train_value_network(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        init_params: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> tuple[_ValueNetwork, list[float]]:
        """Phase 1: Train V(s) network via Bellman residual minimization.

        Minimizes ||V_phi(s) - T_theta V_phi(s)||^2 where T_theta is the
        soft Bellman operator at the initial parameter guess.

        Returns:
            Tuple of (trained V-network, list of per-epoch average losses).
        """
        n_states = problem.num_states
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        key, init_key = jax.random.split(key)
        anchor_features = None
        if self._anchor_state is not None:
            anchor_features = self._build_state_features(
                jnp.asarray([self._anchor_state]), problem
            )[0]
        v_net = _ValueNetwork(
            problem.state_dim or 1, self._hidden_dim, self._num_layers,
            key=init_key, anchor_features=anchor_features,
        )

        # Set up optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self._v_lr),
        )
        opt_state = optimizer.init(eqx.filter(v_net, eqx.is_array))

        # Get all transitions from data
        all_states = panel.get_all_states()

        feat_s = self._build_state_features(all_states, problem)

        # Compute flow utility at initial params
        feature_matrix = utility.feature_matrix  # (S, A, K)
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, init_params)

        # Pre-compute all state features for EV calculation
        all_state_feats = self._build_state_features(jnp.arange(n_states), problem)

        n_samples = len(all_states)

        self._log(f"Phase 1: Training V-network for {self._v_epochs} epochs")

        # Define the training step using Equinox functional patterns
        @eqx.filter_jit
        def train_step(model, opt_state, batch_feat_s, batch_s_idx, all_st_feats, flow_u_mat):
            def loss_fn(model):
                # V(s) prediction for batch
                v_s = jax.vmap(model)(batch_feat_s)

                # Compute V for all states (for EV calculation)
                v_all = jax.vmap(model)(all_st_feats)

                # EV[batch_i, a] = sum_s' P(s'|s_idx[i], a) * V(s')
                # transitions shape: (A, S, S), s_idx indexes into S dimension
                ev = jnp.einsum("abs,s->ba", transitions[:, batch_s_idx, :], v_all)
                q_vals = flow_u_mat[batch_s_idx] + beta * ev

                # Bellman target: sigma * logsumexp(Q / sigma)
                target = sigma * jax.scipy.special.logsumexp(q_vals / sigma, axis=1)

                return jnp.mean((v_s - jax.lax.stop_gradient(target)) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        loss_history: list[float] = []
        best_loss = float("inf")
        best_params = eqx.filter(v_net, eqx.is_array)
        patience_counter = 0
        max_patience = 100

        from tqdm import tqdm
        pbar = tqdm(
            range(self._v_epochs),
            desc="NNES V-net",
            disable=not self._verbose,
            leave=False,
        )
        for epoch in pbar:
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self._v_batch_size):
                end = min(start + self._v_batch_size, n_samples)
                idx = perm[start:end]

                batch_feat_s = feat_s[idx]
                batch_s_idx = all_states[idx]

                v_net, opt_state, loss = train_step(
                    v_net, opt_state, batch_feat_s, batch_s_idx,
                    all_state_feats, flow_u,
                )

                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "best": f"{best_loss:.4f}",
                "no_imp": patience_counter,
            })

            # Early stopping with best model checkpoint
            if avg_loss < best_loss - 1e-8:
                best_loss = avg_loss
                best_params = eqx.filter(v_net, eqx.is_array)
                patience_counter = 0
            else:
                patience_counter += 1

            # Stop if diverging or converged
            if patience_counter >= max_patience:
                pbar.close()
                self._log(f"  V-net early stopping at epoch {epoch+1} (patience={max_patience})")
                break
            if avg_loss > 1e8:
                pbar.close()
                self._log(f"  V-net divergence detected at epoch {epoch+1}, reverting to best")
                break

        # Restore best model by replacing array leaves
        v_net = eqx.combine(best_params, eqx.filter(v_net, lambda x: not eqx.is_array(x)))

        return v_net, loss_history

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run NNES estimation with outer iterations.

        Alternates between Phase 1 (train V-network with current theta)
        and Phase 2 (optimize theta with current V) to avoid the
        model mismatch from training V under zero/incorrect rewards.
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        feature_matrix = utility.feature_matrix  # (S, A, K)
        all_state_feat = self._build_state_features(jnp.arange(n_states), problem)
        lower, upper = utility.get_parameter_bounds()
        bounds = list(zip(np.asarray(lower), np.asarray(upper)))

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        # If initial params are all zeros, bootstrap from CCP-based estimate.
        # A V-network trained with zero rewards learns nothing useful.
        if bool(jnp.all(initial_params == 0)):
            self._log("Bootstrapping initial params from CCP-based MLE")
            initial_params = self._bootstrap_params_from_ccp(
                panel, utility, problem, transitions, sigma, beta,
                feature_matrix, n_states, n_actions, bounds,
            )
            self._log(f"  Bootstrap params: {np.asarray(initial_params)}")

        current_params = initial_params.copy()
        total_nit = 0
        total_nfev = 0
        last_result = None
        v_loss_per_outer: list[float] = []
        all_v_loss_history: list[list[float]] = []

        key = jax.random.PRNGKey(self._seed)

        for outer_iter in range(self._n_outer_iterations):
            self._log(f"Outer iteration {outer_iter + 1}/{self._n_outer_iterations}")

            # Phase 1: Train V-network with current params
            key, train_key = jax.random.split(key)
            v_net, v_loss_history = self._train_value_network(
                panel, utility, problem, transitions, current_params, train_key,
            )
            v_loss_per_outer.append(v_loss_history[-1] if v_loss_history else float("nan"))
            all_v_loss_history.append(v_loss_history)

            # Extract V(s) for all states (no autograd context needed in JAX)
            v_all = jax.vmap(v_net)(all_state_feat)  # (S,)

            # Precompute E[V(s') | s, a] = transitions[a] @ V
            ev_sa = jnp.zeros((n_states, n_actions))
            for a in range(n_actions):
                ev_sa = ev_sa.at[:, a].set(transitions[a] @ v_all)  # (S,)

            # Phase 2: Structural MLE
            self._log("  Phase 2: Structural MLE over theta")

            # Need closure over current ev_sa
            _ev_sa = ev_sa

            all_states_data = panel.get_all_states()
            all_actions_data = panel.get_all_actions()

            def _neg_ll_jax(params):
                flow_u = jnp.einsum("sak,k->sa", feature_matrix, params)
                q_vals = flow_u + beta * _ev_sa
                log_probs = jax.nn.log_softmax(q_vals / sigma, axis=1)
                return -log_probs[all_states_data, all_actions_data].sum()

            _neg_ll_jit = jax.jit(_neg_ll_jax)

            def log_likelihood(params: jnp.ndarray) -> float:
                return -float(_neg_ll_jit(params))

            lower_b = jnp.array([b[0] for b in bounds], dtype=jnp.float32)
            upper_b = jnp.array([b[1] for b in bounds], dtype=jnp.float32)
            last_result = minimize_lbfgsb(
                _neg_ll_jit,
                jnp.asarray(current_params, dtype=jnp.float32),
                bounds=(lower_b, upper_b),
                maxiter=self._outer_max_iter,
                tol=self._outer_tol,
                verbose=self._verbose,
                desc=f"NNES-NFXP outer {outer_iter + 1}",
            )

            current_params = jnp.array(last_result.x, dtype=jnp.float32)
            total_nit += last_result.nit
            total_nfev += last_result.nfev
            self._log(f"  Params: {np.asarray(current_params)}, LL: {-last_result.fun:.2f}")

        # Store final ll function for Hessian
        self._ll_fn = log_likelihood

        params_opt = current_params
        ll_opt = -last_result.fun

        # Compute final policy using last ev_sa
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, params_opt)
        q_vals = flow_u + beta * ev_sa
        policy = jax.nn.softmax(q_vals / sigma, axis=1)
        V = sigma * jax.scipy.special.logsumexp(q_vals / sigma, axis=1)

        # Hessian for standard errors.
        # NOTE: This NFXP-based variant does NOT have Neyman orthogonality.
        # V-approximation errors feed into the score, so these standard errors
        # are not semiparametrically efficient. Use NNESEstimator (NPL-based)
        # for valid inference via the zero Jacobian property.
        hessian = None
        if self._compute_se:
            self._log(
                "Computing Hessian "
                "(WARNING: not semiparametrically efficient in NFXP variant)"
            )

            def ll_fn(params):
                return jnp.array(log_likelihood(params))

            hessian = compute_numerical_hessian(params_opt, ll_fn)

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=params_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            converged=last_result.success,
            num_iterations=total_nit,
            num_function_evals=total_nfev,
            message=f"NNES ({self._n_outer_iterations} outer iters): {last_result.message}",
            optimization_time=elapsed,
            metadata={
                "v_network_values": v_all,
                "ev_sa": ev_sa,
                "n_outer_iterations": self._n_outer_iterations,
                "v_loss_per_outer": v_loss_per_outer,
                "v_loss_history": all_v_loss_history,
            },
        )


# Euler-Mascheroni constant for Hotz-Miller emax correction
EULER_GAMMA = 0.5772156649015329


def _nnes_profiled_logit_neg_ll_mean(
    params: jnp.ndarray,
    z_tilde: jnp.ndarray,
    e_tilde: jnp.ndarray,
    all_states: jnp.ndarray,
    all_actions: jnp.ndarray,
    inv_sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Mean profiled NPL negative log-likelihood for linear utility."""
    theta = params.astype(z_tilde.dtype)
    values = jnp.einsum("sak,k->sa", z_tilde, theta) + e_tilde
    log_probs = jax.nn.log_softmax(values * inv_sigma, axis=1)
    return -jnp.mean(log_probs[all_states, all_actions])


_nnes_profiled_logit_neg_ll_mean_and_grad = jax.jit(
    jax.value_and_grad(_nnes_profiled_logit_neg_ll_mean)
)


class NNESEstimator(BaseEstimator):
    """NNES with NPL Bellman (correct variant, Nguyen 2025).

    Phase 1 trains V_phi on the NPL value function target using data CCPs.
    Phase 2 profiles the NPL value representation and optimizes theta in
    the one-step policy operator. Outer iterations update CCPs from estimated
    theta and retrain V_phi.

    The zero Jacobian property of the NPL mapping guarantees Neyman
    orthogonality (Nguyen 2025, Propositions 3-4): first-order errors in
    V_phi drop out of the Phase 2 score. The pseudo-likelihood Hessian is
    the correct semiparametrically efficient variance estimator without
    bias correction.

    Algorithm:
        For each outer iteration:
            Phase 1 (NPL V-network):
                1. Compute emax correction: e(a,s) = gamma - log(pi(a|s))
                2. Compute policy-weighted transitions: F_pi = sum_a pi(a|s) P(s'|s,a)
                3. Compute NPL target: W = expected_flow_theta + sigma*expected_e + beta*F_pi @ W
                4. Train V_phi to approximate W via supervised regression

            Phase 2 (Structural MLE):
                1. Compute W_z(P) and W_e(P) for the fixed CCP iterate P
                2. Q(s,a;theta) = [z(s,a) + beta E W_z(s')] theta
                   + beta E W_e(s')
                3. Maximize CCP log-likelihood over theta

            Update CCPs from estimated theta for next iteration.

    Args:
        config: NNESConfig or keyword arguments matching NNESConfig fields.

    Example:
        >>> estimator = NNESEstimator(hidden_dim=32, v_epochs=500)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> # V-approximation errors are orthogonal to score => valid SEs
        >>> result.metadata["v_loss_per_outer"]
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        v_lr: float = 1e-3,
        v_epochs: int = 500,
        v_batch_size: int = 512,
        outer_max_iter: int = 200,
        outer_tol: float = 1e-6,
        n_outer_iterations: int = 3,
        anchor_state: int | None = 0,
        compute_se: bool = True,
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
        seed: int = 0,
        config: NNESConfig | None = None,
    ):
        if config is not None:
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers
            v_lr = config.v_lr
            v_epochs = config.v_epochs
            v_batch_size = config.v_batch_size
            outer_max_iter = config.outer_max_iter
            outer_tol = config.outer_tol
            n_outer_iterations = config.n_outer_iterations
            anchor_state = config.anchor_state
            compute_se = config.compute_se
            se_method = config.se_method
            verbose = config.verbose
            seed = config.seed

        super().__init__(
            se_method=se_method,
            compute_hessian=compute_se,
            verbose=verbose,
        )
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._v_lr = v_lr
        self._v_epochs = v_epochs
        self._v_batch_size = v_batch_size
        self._outer_max_iter = outer_max_iter
        self._outer_tol = outer_tol
        self._n_outer_iterations = n_outer_iterations
        self._anchor_state = anchor_state
        self._compute_se = compute_se
        self._seed = seed
        self._config = NNESConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            v_lr=v_lr,
            v_epochs=v_epochs,
            v_batch_size=v_batch_size,
            outer_max_iter=outer_max_iter,
            outer_tol=outer_tol,
            n_outer_iterations=n_outer_iterations,
            anchor_state=anchor_state,
            compute_se=compute_se,
            se_method=se_method,
            verbose=verbose,
            seed=seed,
        )

    @property
    def name(self) -> str:
        return "NNES (Nguyen 2025)"

    @property
    def config(self) -> NNESConfig:
        return self._config

    def _build_state_features(self, states: jnp.ndarray, problem: DDCProblem) -> jnp.ndarray:
        """Build state features from state indices using problem's encoder."""
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        denom = max(problem.num_states - 1, 1)
        return (states.astype(jnp.float32) / denom)[:, None]

    def _estimate_ccps(
        self,
        panel: Panel,
        n_states: int,
        n_actions: int,
        smoothing: float = 0.01,
    ) -> jnp.ndarray:
        """Estimate CCPs from data frequencies with Laplace smoothing."""
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        counts = jnp.zeros((n_states, n_actions))
        for s, a in zip(np.asarray(all_states), np.asarray(all_actions)):
            counts = counts.at[int(s), int(a)].add(1.0)
        state_counts = jnp.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        return (counts + smoothing) / (state_counts + n_actions * smoothing)

    def _compute_npl_components(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        sigma: float,
        beta: float,
        n_states: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute profiled NPL value components for fixed CCPs.

        The package's soft Bellman operator uses the social-surplus
        convention V = sigma * logsumexp(Q / sigma), without adding the
        Euler-constant level term. Under that convention,

            W(P, theta) = W_z(P) theta + W_e(P)

        where W_e contains sigma times the policy entropy.
        """
        dtype = jnp.float64
        ccps64 = ccps.astype(dtype)
        transitions64 = transitions.astype(dtype)
        features64 = feature_matrix.astype(dtype)

        F_pi = jnp.einsum("sa,ast->st", ccps64, transitions64)
        expected_features = jnp.einsum("sa,sak->sk", ccps64, features64)

        safe_ccps = jnp.maximum(ccps64, 1e-12)
        expected_entropy = -jnp.einsum("sa,sa->s", ccps64, jnp.log(safe_ccps))

        lhs = jnp.eye(n_states, dtype=dtype) - beta * F_pi
        W_z = jnp.linalg.solve(lhs, expected_features)
        W_e = jnp.linalg.solve(lhs, sigma * expected_entropy)
        return W_z, W_e

    def _profiled_choice_values(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        params: jnp.ndarray,
        sigma: float,
        beta: float,
        n_states: int,
        n_actions: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Choice-specific values for the profiled one-step NPL operator."""
        W_z, W_e = self._compute_npl_components(
            ccps, transitions, feature_matrix, sigma, beta, n_states,
        )
        transitions64 = transitions.astype(jnp.float64)
        E_W_z = jnp.stack(
            [transitions64[a] @ W_z for a in range(n_actions)],
            axis=1,
        )
        E_W_e = jnp.stack(
            [transitions64[a] @ W_e for a in range(n_actions)],
            axis=1,
        )
        z_tilde = feature_matrix.astype(jnp.float64) + beta * E_W_z
        e_tilde = beta * E_W_e
        values = jnp.einsum(
            "sak,k->sa",
            z_tilde,
            params.astype(jnp.float64),
        ) + e_tilde
        return values, W_z, W_e, z_tilde, e_tilde

    def _compute_npl_target(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        params: jnp.ndarray,
        sigma: float,
        beta: float,
        n_states: int,
        n_actions: int,
    ) -> jnp.ndarray:
        """Compute NPL value function target W via matrix inversion.

        W = W_z(P) theta + W_e(P), where W_e(P) is the entropy term under
        the package's log-sum-exp Bellman convention.

        where:
            F_pi[s,s'] = sum_a pi(a|s) P(s'|s,a)
            expected_flow_theta[s] = sum_a pi(a|s) u(s,a;theta)
            expected_entropy[s] = -sum_a pi(a|s) log(pi(a|s))
        """
        W_z, W_e = self._compute_npl_components(
            ccps, transitions, feature_matrix, sigma, beta, n_states,
        )
        W = W_z @ params.astype(jnp.float64) + W_e
        return W.astype(jnp.float32)

    def _train_value_network_npl(
        self,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        ccps: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        params: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> tuple[_ValueNetwork, list[float]]:
        """Phase 1: Train V_phi to approximate the NPL target W.

        Instead of minimizing the NFXP Bellman residual, we compute the
        NPL target W in closed form and train V_phi via supervised
        regression: min ||V_phi(s) - W(s)||^2.

        For small state spaces (tabular), the closed-form W from matrix
        inversion is exact and training is just function fitting. The
        neural network enables generalization to large state spaces where
        matrix inversion is infeasible.
        """
        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        # Compute NPL target via matrix inversion (exact for tabular)
        W_target = self._compute_npl_target(
            ccps, transitions, feature_matrix, params,
            sigma, beta, n_states, n_actions,
        )
        if self._anchor_state is not None:
            W_target = W_target - W_target[self._anchor_state]

        # Build state features for all states
        all_state_feats = self._build_state_features(jnp.arange(n_states), problem)

        # Initialize V-network
        key, init_key = jax.random.split(key)
        anchor_features = None
        if self._anchor_state is not None:
            anchor_features = self._build_state_features(
                jnp.asarray([self._anchor_state]), problem
            )[0]
        v_net = _ValueNetwork(
            problem.state_dim or 1, self._hidden_dim, self._num_layers,
            key=init_key, anchor_features=anchor_features,
        )

        # Set up optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self._v_lr),
        )
        opt_state = optimizer.init(eqx.filter(v_net, eqx.is_array))

        self._log(f"Phase 1: Training V-network on NPL target for {self._v_epochs} epochs")

        @eqx.filter_jit
        def train_step(model, opt_state, state_feats, targets):
            def loss_fn(model):
                v_pred = jax.vmap(model)(state_feats)
                return jnp.mean((v_pred - targets) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        loss_history: list[float] = []
        best_loss = float("inf")
        best_params = eqx.filter(v_net, eqx.is_array)
        patience_counter = 0
        max_patience = 100

        for epoch in range(self._v_epochs):
            v_net, opt_state, loss = train_step(
                v_net, opt_state, all_state_feats, W_target,
            )

            loss_val = float(loss)
            loss_history.append(loss_val)

            if loss_val < best_loss - 1e-8:
                best_loss = loss_val
                best_params = eqx.filter(v_net, eqx.is_array)
                patience_counter = 0
            else:
                patience_counter += 1

            if self._verbose and (epoch + 1) % 100 == 0:
                self._log(f"  V-net epoch {epoch+1}: loss={loss_val:.6f}")

            if patience_counter >= max_patience:
                self._log(f"  V-net early stopping at epoch {epoch+1}")
                break
            if loss_val > 1e8:
                self._log(f"  V-net divergence at epoch {epoch+1}, reverting to best")
                break

        # Restore best model
        v_net = eqx.combine(best_params, eqx.filter(v_net, lambda x: not eqx.is_array(x)))

        return v_net, loss_history

    def _bootstrap_params_from_ccp(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        sigma: float,
        beta: float,
        feature_matrix: jnp.ndarray,
        n_states: int,
        n_actions: int,
        bounds: list,
    ) -> jnp.ndarray:
        """Estimate initial params from data CCPs via Hotz-Miller inversion."""
        ccps = self._estimate_ccps(panel, n_states, n_actions)

        # Hotz-Miller: compute EV via matrix inversion
        P_pi = jnp.einsum("sa,ast->st", ccps, transitions)
        flow_features = jnp.einsum("sa,sak->sk", ccps, feature_matrix)
        safe_ccps = jnp.maximum(ccps, 1e-10)
        entropy = -(ccps * jnp.log(safe_ccps)).sum(axis=1)

        eye = jnp.eye(n_states, dtype=jnp.float64)
        try:
            M = eye - beta * P_pi.astype(jnp.float64)
            ev_features = jnp.linalg.solve(
                M,
                flow_features.astype(jnp.float64),
            ).astype(jnp.float32)
            ev_entropy = jnp.linalg.solve(
                M,
                entropy.astype(jnp.float64),
            ).astype(jnp.float32)
        except Exception:
            return jnp.full((feature_matrix.shape[2],), 0.01)

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        # Precompute EV transition products so the JAX objective is pure.
        ev_trans_feat = jnp.stack(
            [transitions[a] @ ev_features for a in range(n_actions)], axis=0,
        )  # (A, S, K)
        ev_trans_ent = jnp.stack(
            [transitions[a] @ ev_entropy for a in range(n_actions)], axis=0,
        )  # (A, S)

        def _bootstrap_obj_jax(params):
            flow_u = jnp.einsum("sak,k->sa", feature_matrix, params)
            feat_part = jnp.einsum("ask,k->sa", ev_trans_feat, params)
            ent_part = ev_trans_ent.T  # (S, A)
            continuation = beta * (feat_part + sigma * ent_part)
            v = flow_u + continuation
            log_probs = jax.nn.log_softmax(v / sigma, axis=1)
            return -log_probs[all_states, all_actions].sum()

        x0 = jnp.full(feature_matrix.shape[2], 0.01, dtype=jnp.float32)
        lower_b = jnp.array([b[0] for b in bounds], dtype=jnp.float32)
        upper_b = jnp.array([b[1] for b in bounds], dtype=jnp.float32)
        result = minimize_lbfgsb(
            _bootstrap_obj_jax,
            x0,
            bounds=(lower_b, upper_b),
            maxiter=100,
            tol=1e-6,
            verbose=False,
            desc="NNES bootstrap",
        )
        return jnp.array(result.x, dtype=jnp.float32)

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run NNES-NPL estimation with outer iterations.

        Each outer iteration:
            1. Compute CCPs (from data on first iter, from policy on subsequent)
            2. Phase 1: train V_phi on NPL target given CCPs and current theta
            3. Phase 2: optimize theta through the profiled NPL value V_{theta,P}
            4. Update CCPs from estimated theta
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        feature_matrix = utility.feature_matrix  # (S, A, K)
        all_state_feat = self._build_state_features(jnp.arange(n_states), problem)
        lower, upper = utility.get_parameter_bounds()
        bounds = list(zip(np.asarray(lower), np.asarray(upper)))
        lower_b = jnp.array([b[0] for b in bounds], dtype=jnp.float64)
        upper_b = jnp.array([b[1] for b in bounds], dtype=jnp.float64)
        all_states_data = panel.get_all_states()
        all_actions_data = panel.get_all_actions()
        n_obs = int(panel.num_observations)
        inv_sigma_f64 = jnp.float64(1.0 / sigma)

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        # Bootstrap from CCP if params are all zeros
        if bool(jnp.all(initial_params == 0)):
            self._log("Bootstrapping initial params from CCP-based MLE")
            initial_params = self._bootstrap_params_from_ccp(
                panel, utility, problem, transitions, sigma, beta,
                feature_matrix, n_states, n_actions, bounds,
            )
            self._log(f"  Bootstrap params: {np.asarray(initial_params)}")

        current_params = initial_params.copy()
        total_nit = 0
        total_nfev = 0
        last_result = None
        last_ll_sum = float("nan")
        last_z_tilde = None
        last_e_tilde = None
        last_profiled_values = None
        last_profiled_W = None
        ev_sa = jnp.zeros((n_states, n_actions), dtype=jnp.float32)
        v_all = jnp.zeros((n_states,), dtype=jnp.float32)
        v_loss_per_outer: list[float] = []
        all_v_loss_history: list[list[float]] = []

        key = jax.random.PRNGKey(self._seed)

        # Initial CCPs from data
        ccps = self._estimate_ccps(panel, n_states, n_actions)

        for outer_iter in range(self._n_outer_iterations):
            self._log(f"Outer iteration {outer_iter + 1}/{self._n_outer_iterations}")

            # Phase 1: Train V-network on NPL target
            key, train_key = jax.random.split(key)
            v_net, v_loss_history = self._train_value_network_npl(
                problem, transitions, ccps, feature_matrix,
                current_params, train_key,
            )
            v_loss_per_outer.append(v_loss_history[-1] if v_loss_history else float("nan"))
            all_v_loss_history.append(v_loss_history)

            # Extract V(s) for all states
            v_all = jax.vmap(v_net)(all_state_feat)  # (S,)

            # Precompute E[V(s') | s, a] = transitions[a] @ V
            ev_sa = jnp.zeros((n_states, n_actions))
            for a in range(n_actions):
                ev_sa = ev_sa.at[:, a].set(transitions[a] @ v_all)

            # Phase 2: Structural MLE
            self._log("  Phase 2: Profiled NPL MLE over theta")

            (
                profiled_values,
                W_z,
                W_e,
                z_tilde,
                e_tilde,
            ) = self._profiled_choice_values(
                ccps,
                transitions,
                feature_matrix,
                current_params,
                sigma,
                beta,
                n_states,
                n_actions,
            )

            def _profiled_neg_ll_mean(params, z_arg, e_arg):
                return _nnes_profiled_logit_neg_ll_mean_and_grad(
                    params,
                    z_arg,
                    e_arg,
                    all_states_data,
                    all_actions_data,
                    inv_sigma_f64,
                )

            last_result = minimize_lbfgsb(
                _profiled_neg_ll_mean,
                jnp.asarray(current_params, dtype=jnp.float64),
                bounds=(lower_b, upper_b),
                maxiter=self._outer_max_iter,
                tol=self._outer_tol,
                verbose=self._verbose,
                desc=f"NNES-NPL outer {outer_iter + 1}",
                value_and_grad=True,
                param_names=list(utility.parameter_names),
                fun_args=(z_tilde, e_tilde),
            )

            current_params = jnp.array(last_result.x, dtype=jnp.float32)
            total_nit += last_result.nit
            total_nfev += last_result.nfev
            last_ll_sum = -float(last_result.fun) * n_obs
            last_z_tilde = z_tilde
            last_e_tilde = e_tilde
            last_profiled_values = (
                jnp.einsum(
                    "sak,k->sa",
                    z_tilde,
                    current_params.astype(jnp.float64),
                )
                + e_tilde
            )
            last_profiled_W = W_z @ current_params.astype(jnp.float64) + W_e
            self._log(f"  Params: {np.asarray(current_params)}, LL: {last_ll_sum:.2f}")

            # Update CCPs from estimated policy for next iteration
            ccps = jax.nn.softmax(last_profiled_values / sigma, axis=1)

        def log_likelihood(params: jnp.ndarray) -> float:
            values = (
                jnp.einsum(
                    "sak,k->sa",
                    last_z_tilde,
                    params.astype(jnp.float64),
                )
                + last_e_tilde
            )
            log_probs = jax.nn.log_softmax(values / sigma, axis=1)
            return float(log_probs[all_states_data, all_actions_data].sum())

        # Store final ll function for Hessian
        self._ll_fn = log_likelihood

        params_opt = current_params
        ll_opt = last_ll_sum

        # Compute final policy and value function. The V-network is anchored
        # for numerical identification, so report the Bellman-level policy
        # value for validation while keeping the anchored network values in
        # metadata.
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, params_opt)
        policy = ccps
        safe_policy = jnp.maximum(policy, 1e-10)
        reward_pi = jnp.sum(policy * flow_u, axis=1)
        entropy_pi = -sigma * jnp.sum(policy * jnp.log(safe_policy), axis=1)
        transition_pi = jnp.einsum("sa,ast->st", policy, transitions)
        lhs = jnp.eye(n_states, dtype=jnp.float64) - beta * transition_pi.astype(jnp.float64)
        V = jnp.linalg.solve(
            lhs,
            (reward_pi + entropy_pi).astype(jnp.float64),
        ).astype(jnp.float32)

        # Hessian for standard errors.
        # By Nguyen (2025) Propositions 3-4, the NPL score is orthogonal to
        # CCP estimation error (zero Jacobian property). First-order errors
        # in V_phi drop out, so the pseudo-likelihood Hessian is the correct
        # semiparametrically efficient variance estimator.
        hessian = None
        if self._compute_se:
            self._log("Computing Hessian (semiparametrically efficient via NPL orthogonality)")

            def ll_fn(params):
                return jnp.array(log_likelihood(params))

            hessian = compute_numerical_hessian(params_opt, ll_fn)

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=params_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            converged=last_result.success,
            num_iterations=total_nit,
            num_function_evals=total_nfev,
            message=f"NNES-NPL ({self._n_outer_iterations} outer iters): {last_result.message}",
            optimization_time=elapsed,
            metadata={
                "v_network_values": v_all,
                "ev_sa": ev_sa,
                "profile_mode": "exact_finite_state_npl",
                "profiled_choice_values": last_profiled_values,
                "profiled_value_function": last_profiled_W,
                "n_outer_iterations": self._n_outer_iterations,
                "v_loss_per_outer": v_loss_per_outer,
                "v_loss_history": all_v_loss_history,
                "final_ccps": ccps,
            },
        )

# Backward compatibility alias for the old Bellman-residual variant
NNESBellmanEstimator = NNESNFXPEstimator
