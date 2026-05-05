"""TD-CCP Estimator for dynamic discrete choice models.

Implements the Temporal-Difference CCP algorithm from Adusumilli and Eckardt
(2025). The paper proposes two methods for estimating the recursive terms
h(a,x) and g(a,x) in the CCP pseudo-log-likelihood without requiring
estimation of transition densities K(x'|a,x):

1. Linear semi-gradient: closed-form matrix solve using basis functions.
   Fast (sub-second) and recommended for moderate-dimensional problems.
   Implements equation (3.5) from the paper.

2. Approximate Value Iteration (AVI): iterative regression using any
   ML method (here, neural networks). More flexible for high-dimensional
   state spaces. Implements Algorithm 1 from the paper.

Both methods learn h(a,x) and g(a,x) directly from observed (a,x,a',x')
transitions. The paper's key contribution is that transition densities are
never needed because the TD fixed-point characterization lets us estimate
h and g from sample expectations over successive state-action pairs.

For valid inference, the paper introduces a locally robust moment condition
(equation 4.6) that corrects for first-stage nonparametric estimation error
in h and g. This requires cross-fitting (Algorithm 2) and a backward value
function lambda (Algorithm 5). Without these corrections, standard errors
from naive plug-in MLE understate uncertainty.

References:
    Adusumilli, K. and Eckardt, D. (2025). "Temporal-Difference Estimation
        of Dynamic Discrete Choice Models."
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Tsitsiklis, J.N. and Van Roy, B. (1997). "An Analysis of Temporal-
        Difference Learning with Function Approximation." IEEE TAC 42(5).
    Munos, R. and Szepesvari, C. (2008). "Finite-Time Bounds for Fitted
        Value Iteration." JMLR 9, 815-857.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TDCCPConfig:
    """Configuration for the TD-CCP Estimator.

    The two main algorithmic choices are controlled by ``method`` and
    ``cross_fitting``. The method selects between the linear semi-gradient
    (fast closed-form solve) and neural AVI (flexible but slower). Cross-
    fitting enables valid inference at parametric rates per the paper's
    Algorithm 2.

    Attributes:
        method: Which TD method to use for estimating h and g.
            "semigradient" uses the linear semi-gradient (eq 3.5), which
            is a single matrix inversion per component. Fast and recommended
            for tabular or moderate-dimensional state spaces.
            "neural" uses neural AVI (Algorithm 1), which iteratively trains
            neural networks. More flexible for continuous/high-dimensional
            state spaces.
        basis_dim: Basis size control for the semi-gradient method. For
            ``basis_type="polynomial"``, this is the scalar state polynomial
            degree used by the historical tabular path. For
            ``basis_type="encoded"``, this is the maximum elementwise
            polynomial degree applied to ``problem.state_encoder`` features.
            Ignored when method="neural" or ``basis_type="tabular"``.
        basis_type: Semi-gradient basis construction. ``"polynomial"``
            preserves the original scalar state-index basis. ``"encoded"``
            uses ``problem.state_encoder`` features and action interactions,
            which is the right path for high-dimensional encoded-state DGPs.
            ``"tabular"`` uses one indicator per state-action pair.
        basis_include_rewards: Whether the encoded basis also includes the
            known utility feature vector z(a, x). This is useful when reward
            features contain nonlinear functions of the encoded state.
        basis_ridge: Ridge stabilization added to the semi-gradient normal
            equation. This should be small for well-covered bases and larger
            for high-dimensional or nearly collinear encoded bases.
        basis_pinv_rcond: If positive, solve the semi-gradient normal equation
            with a Moore-Penrose pseudoinverse using this cutoff instead of a
            direct solve. This stabilizes nearly singular high-dimensional
            bases by dropping weak empirical directions.
        hidden_dim: Number of units in each hidden layer of the MLP.
            Only used when method="neural".
        num_hidden_layers: Number of hidden layers in the MLP.
            Only used when method="neural".
        avi_iterations: Number of approximate value iteration rounds.
            Paper recommends J=20. Each round freezes the target network
            and fits the current network to the new targets. Corresponds
            to the outer loop in Algorithm 1.
        avi_early_stop_tol: Relative change threshold for early stopping
            of AVI iterations. Monitors epsilon_j from footnote 9 of the
            paper: epsilon_j = E_n[||h_{j+1} - h_j||^2] / E_n[||h_j - E[h_j]||^2].
            Stops when epsilon_j < this value. Set to 0.0 to disable.
        epochs_per_avi: Number of SGD epochs per AVI iteration.
            Only used when method="neural".
        learning_rate: Learning rate for neural network training (ADAM).
            Only used when method="neural".
        batch_size: Mini-batch size for SGD.
            Only used when method="neural".
        ccp_method: How to estimate first-stage CCPs.
            "frequency" uses frequency counting with additive smoothing.
            "logit" uses logistic regression with polynomial features,
            as recommended by the paper for continuous state spaces.
        ccp_smoothing: Smoothing constant for frequency-based CCP estimation.
            Added to each count to prevent log(0). Only used when
            ccp_method="frequency".
        ccp_poly_degree: Polynomial degree for logit CCP estimation.
            Only used when ccp_method="logit". Paper recommends 2-3.
        cross_fitting: Whether to use 2-fold cross-fitting for valid
            inference at parametric rates (Algorithm 2 in the paper).
            When True, data is split into two folds. Fold 1 estimates h,g.
            Fold 2 uses those estimates for parameter estimation (and vice
            versa). The two theta estimates are averaged. This is required
            for the locally robust moment to achieve sqrt(n) consistency.
        robust_se: Whether to compute locally robust standard errors
            using the backward value function lambda and the debiased
            moment condition zeta (Section 4 of the paper). When False,
            uses the naive numerical Hessian of the plug-in log-likelihood,
            which understates uncertainty by ignoring first-stage error.
            Requires cross_fitting=True to be valid.
        n_policy_iterations: Number of NPL-style outer iterations.
            The paper does NOT use policy iteration. It estimates h,g once
            and plugs into the pseudo-log-likelihood. Setting this to 1
            matches the paper. Values > 1 add an NPL refinement loop that
            re-solves the Bellman equation (which the paper avoids).
            Default is 1 to match the paper.
        policy_iteration_tol: Convergence tolerance for NPL iterations.
            Only used when n_policy_iterations > 1.
        outer_max_iter: Maximum L-BFGS-B iterations for partial MLE.
        outer_tol: Gradient tolerance for L-BFGS-B convergence.
        theta_l2_penalty: Optional L2 penalty on structural parameters in
            the partial PMLE. Defaults to zero, matching the paper. This is
            a finite-sample stabilization knob for high-dimensional reward
            specifications.
        compute_se: Whether to compute standard errors at all.
        verbose: Whether to print progress messages.
    """

    # --- Method selection ---
    method: Literal["semigradient", "neural"] = "semigradient"

    # --- Semi-gradient specific ---
    basis_dim: int = 8
    basis_type: Literal["polynomial", "encoded", "tabular"] = "polynomial"
    basis_include_rewards: bool = False
    basis_ridge: float = 1e-8
    basis_pinv_rcond: float | None = None

    # --- Neural AVI specific ---
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    avi_iterations: int = 20
    avi_early_stop_tol: float = 0.01
    epochs_per_avi: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 8192

    # --- CCP estimation ---
    ccp_method: Literal["frequency", "logit"] = "frequency"
    ccp_smoothing: float = 0.01
    ccp_poly_degree: int = 3

    # --- Inference ---
    cross_fitting: bool = True
    split_unit: Literal["individual", "row"] = "individual"
    robust_se: bool = True

    # --- NPL iteration (not in paper, optional) ---
    n_policy_iterations: int = 1
    policy_iteration_tol: float = 1e-4

    # --- Optimizer ---
    outer_max_iter: int = 200
    outer_tol: float = 1e-6
    theta_l2_penalty: float = 0.0
    compute_se: bool = True
    verbose: bool = False


def make_state_action_tabular_utility(
    num_states: int,
    num_actions: int,
    *,
    parameter_prefix: str = "r",
) -> UtilityFunction:
    """Build a saturated state-action reward utility.

    The returned utility has one parameter for each ``(state, action)`` cell.
    Passing a flattened reward matrix in row-major order reconstructs that
    matrix exactly. This is useful for TD-CCP validation cells where the truth
    is a reward matrix rather than a finite low-dimensional theta.
    """
    from econirl.preferences.action_reward import ActionDependentReward

    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")

    num_parameters = num_states * num_actions
    features = jnp.eye(num_parameters, dtype=jnp.float64).reshape(
        num_states,
        num_actions,
        num_parameters,
    )
    parameter_names = [
        f"{parameter_prefix}_{state}_{action}"
        for state in range(num_states)
        for action in range(num_actions)
    ]
    return ActionDependentReward(features, parameter_names)


# ---------------------------------------------------------------------------
# Neural network for AVI method
# ---------------------------------------------------------------------------

class _EVComponentNetwork(eqx.Module):
    """MLP for approximating a single component of h(a,x) or g(a,x).

    Maps (action, state) features to a scalar value. Used in the neural
    AVI method where each AVI iteration fits this network to TD targets.
    """

    mlp: eqx.nn.MLP
    output_shift: jax.Array

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: jax.Array,
        output_shift: float = 0.0,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.relu,
            key=key,
        )
        self.output_shift = jnp.asarray(output_shift, dtype=jnp.float64)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x).squeeze(-1) + self.output_shift


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class TDCCPEstimator(BaseEstimator):
    """TD-CCP Estimator implementing Adusumilli and Eckardt (2025).

    The estimator learns the recursive value function terms h(a,x) and g(a,x)
    from the CCP pseudo-log-likelihood (equation 2.1) using temporal-difference
    methods. Two approaches are available:

    1. Linear semi-gradient (Section 3.1): Closed-form solution for h and g
       using basis function approximation. Solves equation (3.5) directly.

    2. Neural AVI (Section 3.2): Iterative value approximation using neural
       networks. Each AVI round solves a regression problem with TD targets.

    For valid inference, the estimator implements 2-fold cross-fitting
    (Algorithm 2) and locally robust standard errors (Section 4).
    """

    def __init__(
        self,
        config: TDCCPConfig | None = None,
        se_method: SEMethod = "asymptotic",
        seed: int = 0,
        **kwargs,
    ):
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
        return "TD-CCP"

    @property
    def config(self) -> TDCCPConfig:
        return self._config

    # ==================================================================
    # Step 1: CCP estimation
    # ==================================================================
    # The paper recommends logit with polynomial features for continuous
    # states (Section 3, practical recommendations). For discrete tabular
    # problems, frequency counting with smoothing is standard.
    # ==================================================================

    def _estimate_ccps(
        self,
        panel: Panel,
        num_states: int,
        num_actions: int,
    ) -> jnp.ndarray:
        """Estimate conditional choice probabilities P(a|x).

        Two methods are available:
        - "frequency": counts + additive smoothing (standard for tabular).
        - "logit": logistic regression with polynomial features in the
          state variable, as recommended by the paper for continuous states.

        Returns:
            CCP matrix of shape (num_states, num_actions).
        """
        if self._config.ccp_method == "logit":
            return self._estimate_ccps_logit(panel, num_states, num_actions)
        else:
            return self._estimate_ccps_frequency(panel, num_states, num_actions)

    def _estimate_ccps_frequency(
        self, panel: Panel, num_states: int, num_actions: int
    ) -> jnp.ndarray:
        """Frequency-based CCP estimator with additive smoothing.

        P_hat(a|s) = (N(s,a) + smoothing) / (N(s) + A * smoothing)

        This prevents log(0) in the flow computation for g(a,x). The
        smoothing constant should be small (0.001-0.1) to avoid distorting
        the empirical distribution.
        """
        smoothing = self._config.ccp_smoothing
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        counts = jnp.zeros((num_states, num_actions), dtype=jnp.float64)
        counts = counts.at[all_states, all_actions].add(1.0)

        state_counts = counts.sum(axis=1, keepdims=True)
        ccps = (counts + smoothing) / (state_counts + num_actions * smoothing)
        return ccps

    def _estimate_ccps_logit(
        self, panel: Panel, num_states: int, num_actions: int
    ) -> jnp.ndarray:
        """Logit CCP estimator with polynomial features.

        Fits a multinomial logit model P(a|x) = softmax(X_poly @ beta_a)
        where X_poly includes polynomial terms up to ccp_poly_degree in
        the normalized state variable. This gives smooth CCP estimates
        even at states with few observations, as recommended by the paper.
        """
        all_states = np.array(panel.get_all_states())
        all_actions = np.array(panel.get_all_actions())

        # Normalize states to [0, 1] for numerical stability
        x_norm = all_states / max(num_states - 1, 1)

        # Build polynomial features: [1, x, x^2, ..., x^d]
        degree = self._config.ccp_poly_degree
        X_poly = np.column_stack([x_norm ** p for p in range(degree + 1)])

        # Fit logistic regression via iterative reweighted least squares.
        # For binary actions, this is standard logistic regression.
        # For multiple actions, we fit A-1 log-odds relative to action 0.
        if num_actions == 2:
            # Binary logit: P(a=1|x) = sigmoid(X @ beta)
            y_jax = jnp.array((all_actions == 1).astype(np.float64))
            X_jax = jnp.array(X_poly, dtype=jnp.float64)

            def neg_ll_binary(beta):
                logits = X_jax @ beta
                ll = y_jax * logits - jnp.logaddexp(0.0, logits)
                return -ll.sum()

            from econirl.core.optimizer import minimize_lbfgsb
            result = minimize_lbfgsb(neg_ll_binary, jnp.zeros(X_poly.shape[1], dtype=jnp.float64), maxiter=200, tol=1e-6)
            beta = np.asarray(result.x)

            # Predict P(a=1|s) for all states
            x_all = np.arange(num_states) / max(num_states - 1, 1)
            X_all = np.column_stack([x_all ** p for p in range(degree + 1)])
            logits_all = X_all @ beta
            p1 = 1.0 / (1.0 + np.exp(-logits_all))
            ccps = np.column_stack([1.0 - p1, p1])
        else:
            # Multinomial logit for A > 2 actions.
            # Normalize action 0 as reference (beta_0 = 0).
            n_params = X_poly.shape[1] * (num_actions - 1)

            X_jax_m = jnp.array(X_poly, dtype=jnp.float64)
            actions_jax = jnp.array(all_actions)

            def neg_ll_multi(beta_flat):
                beta = beta_flat.reshape(num_actions - 1, X_jax_m.shape[1])
                logits = X_jax_m @ beta.T  # (N, A-1)
                logits_full = jnp.concatenate([jnp.zeros((len(actions_jax), 1)), logits], axis=1)
                log_sum_exp = jax.scipy.special.logsumexp(logits_full, axis=1)
                ll = logits_full[jnp.arange(len(actions_jax)), actions_jax] - log_sum_exp
                return -ll.sum()

            from econirl.core.optimizer import minimize_lbfgsb
            result = minimize_lbfgsb(neg_ll_multi, jnp.zeros(n_params, dtype=jnp.float64), maxiter=200, tol=1e-6)
            beta = np.asarray(result.x).reshape(num_actions - 1, X_poly.shape[1])

            x_all = np.arange(num_states) / max(num_states - 1, 1)
            X_all = np.column_stack([x_all ** p for p in range(degree + 1)])
            logits = X_all @ beta.T  # (S, A-1)
            logits_full = np.column_stack([np.zeros(num_states), logits])
            exp_logits = np.exp(logits_full - logits_full.max(axis=1, keepdims=True))
            ccps = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Clip to avoid exact 0 or 1 (for log safety)
        ccps = np.clip(ccps, 1e-10, 1.0 - 1e-10)
        ccps = ccps / ccps.sum(axis=1, keepdims=True)
        return jnp.array(ccps, dtype=jnp.float64)

    # ==================================================================
    # Step 2: Extract observed transitions from panel data
    # ==================================================================
    # The paper works with observed (a_t, x_t, a_{t+1}, x_{t+1}) tuples.
    # This is the key difference from the old implementation which
    # marginalized over actions using CCPs. Here we use raw transitions.
    # ==================================================================

    @staticmethod
    def _extract_individual_ids(panel: Panel) -> np.ndarray:
        """Return individual index for each transition tuple from _extract_transitions.

        Produces an array of length equal to the number of transitions, where
        entry i is the index of the individual whose trajectory produced transition i.
        Used for clustering the sandwich estimator by individual.
        """
        ids = []
        for ind_idx, traj in enumerate(panel.trajectories):
            n_trans = max(0, len(traj.actions) - 1)
            if n_trans > 0:
                ids.append(np.full(n_trans, ind_idx, dtype=np.int32))
        return np.concatenate(ids) if ids else np.array([], dtype=np.int32)

    @staticmethod
    def _extract_transitions(panel: Panel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract (a_t, x_t, a_{t+1}, x_{t+1}) tuples from panel data.

        These observed transition tuples are the inputs to the TD algorithms.
        The paper uses E_n[f(a,x,a',x')] = (n(T-1))^{-1} sum_i sum_{t=1}^{T-1}
        f(a_it, x_it, a_{i,t+1}, x_{i,t+1}) throughout.

        Returns:
            Tuple of (actions, states, next_actions, next_states), each a
            1D numpy array of the same length (total number of transitions).
        """
        states = np.array(panel.get_all_states())
        actions = np.array(panel.get_all_actions())
        next_states = np.array(panel.get_all_next_states())

        # For next_actions, we need the action taken at time t+1.
        # Panel stores trajectories; we need a_{t+1} for each (s_t, a_t, s_{t+1}).
        # The next_actions are the actions shifted by one period within each
        # individual's trajectory. We reconstruct them from the panel.
        next_actions_list = []
        for traj in panel.trajectories:
            # Each trajectory has states[0..T-1] and actions[0..T-1].
            # Transition t uses (a_t, s_t, a_{t+1}, s_{t+1}) for t=0..T-2.
            if len(traj.actions) > 1:
                next_actions_list.append(np.array(traj.actions[1:]))
        if next_actions_list:
            next_actions = np.concatenate(next_actions_list)
        else:
            next_actions = np.array([], dtype=np.int32)

        # Truncate to match lengths (states/actions from get_all_states
        # already exclude the last period of each trajectory)
        min_len = min(len(states), len(next_actions))
        return (
            actions[:min_len],
            states[:min_len],
            next_actions[:min_len],
            next_states[:min_len],
        )

    # ==================================================================
    # Step 3a: Linear semi-gradient method (Section 3.1)
    # ==================================================================
    # This is the paper's first proposed method. For each component j of
    # h(a,x), the semi-gradient fixed point is (eq 3.4):
    #
    #   omega_hat = [E_n[phi(a,x)(phi(a,x) - beta*phi(a',x'))^T]]^{-1}
    #              * E_n[phi(a,x) * z_j(a,x)]
    #
    # where phi(a,x) are basis functions over (action, state) pairs and
    # z_j(a,x) is the j-th component of the feature vector z(a,x).
    #
    # Similarly for g(a,x), replacing z(a,x) with beta * e(a',x') where
    # e(a,x) = gamma_E - ln P(a|x).
    #
    # This is a SINGLE matrix solve, not iterative. The key insight from
    # Tsitsiklis and Van Roy (1997) is that for linear function classes,
    # the semi-gradient converges to a fixed point that can be computed
    # directly. No neural networks or gradient descent needed.
    # ==================================================================

    def _build_basis_functions(
        self,
        actions: np.ndarray,
        states: np.ndarray,
        num_states: int,
        num_actions: int,
        problem: DDCProblem | None = None,
        feature_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build semi-gradient basis functions phi(a,x).

        The historical default is a scalar polynomial in the discrete state
        index, interacted with action indicators. High-dimensional known-truth
        cells should use ``basis_type="encoded"``, which builds the same
        action interactions from ``problem.state_encoder`` instead of scalar
        labels.

        Returns:
            Basis matrix with one row per observed (action, state) pair.
        """
        cfg = self._config
        n = len(states)
        actions = np.asarray(actions, dtype=np.int64)
        states = np.asarray(states, dtype=np.int64)

        if cfg.basis_type == "tabular":
            phi = np.zeros((n, num_states * num_actions), dtype=np.float64)
            cols = states * num_actions + actions
            phi[np.arange(n), cols] = 1.0
            return phi

        if cfg.basis_type == "encoded":
            if problem is not None and problem.state_encoder is not None:
                state_feats = np.asarray(problem.state_encoder(jnp.asarray(states)), dtype=np.float64)
            else:
                denom = max(num_states - 1, 1)
                state_feats = (states.astype(np.float64) / denom)[:, None]
            if state_feats.ndim == 1:
                state_feats = state_feats[:, None]

            degree = max(int(cfg.basis_dim), 1)
            if problem is not None and cfg.basis_dim >= num_states:
                all_states = jnp.arange(num_states, dtype=jnp.int32)
                if problem.state_encoder is not None:
                    centers = np.asarray(problem.state_encoder(all_states), dtype=np.float64)
                else:
                    denom = max(num_states - 1, 1)
                    centers = (np.arange(num_states, dtype=np.float64) / denom)[:, None]
                if centers.ndim == 1:
                    centers = centers[:, None]
                center_diff = centers[:, None, :] - centers[None, :, :]
                center_sqdist = np.sum(center_diff * center_diff, axis=2)
                center_dist = np.sqrt(center_sqdist)
                positive_dist = np.where(center_dist > 1e-12, center_dist, np.inf)
                nearest_dist = np.min(positive_dist, axis=1)
                nearest_dist = nearest_dist[np.isfinite(nearest_dist)]
                bandwidth = float(np.median(nearest_dist)) if nearest_dist.size else 1.0
                bandwidth = max(bandwidth, 1e-3)
                diff = state_feats[:, None, :] - centers[None, :, :]
                sqdist = np.sum(diff * diff, axis=2)
                blocks = [np.exp(-0.5 * sqdist / (bandwidth * bandwidth))]
            else:
                blocks = [np.ones((n, 1), dtype=np.float64)]
                for p in range(1, degree + 1):
                    blocks.append(np.power(state_feats, p))
            if cfg.basis_include_rewards and feature_matrix is not None:
                z_rows = np.asarray(feature_matrix[states], dtype=np.float64)
                z_rows = z_rows.reshape(n, -1)
                blocks.append(z_rows)
            state_basis = np.concatenate(blocks, axis=1)

            n_basis_per_action = state_basis.shape[1]
            total_basis = num_actions * n_basis_per_action
            phi = np.zeros((n, total_basis), dtype=np.float64)
            for a in range(num_actions):
                mask = actions == a
                offset = a * n_basis_per_action
                phi[mask, offset : offset + n_basis_per_action] = state_basis[mask]
            return phi

        # Normalize states to [0, 1]
        x_norm = states / max(num_states - 1, 1)

        # Polynomial features for each action
        # phi(a, x) = indicator(action=a) * [1, x, x^2, ..., x^d]
        d = int(cfg.basis_dim)
        n_basis_per_action = d + 1
        total_basis = num_actions * n_basis_per_action

        phi = np.zeros((n, total_basis), dtype=np.float64)
        for a in range(num_actions):
            mask = (actions == a)
            offset = a * n_basis_per_action
            for p in range(n_basis_per_action):
                phi[mask, offset + p] = x_norm[mask] ** p

        return phi

    def _semigradient_solve(
        self,
        actions: np.ndarray,
        states: np.ndarray,
        next_actions: np.ndarray,
        next_states: np.ndarray,
        feature_matrix: np.ndarray,
        ccps: np.ndarray,
        num_states: int,
        num_actions: int,
        gamma: float,
        problem: DDCProblem | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate h(a,x) and g(a,x) via linear semi-gradient (eq 3.5, 3.6).

        This implements the core of the paper's first method. For each
        scalar component h^{(j)} of h, we solve:

            omega^{(j)} = A^{-1} * b^{(j)}

        where:
            A = E_n[phi(a,x) * (phi(a,x) - beta * phi(a',x'))^T]
            b^{(j)} = E_n[phi(a,x) * z_j(a,x)]

        The matrix A is shared across all components of h (and g), so it
        only needs to be inverted once. This is the key computational
        advantage of the semi-gradient method.

        For g, we replace z(a,x) with beta * e(a',x') where
        e(a,x) = euler_constant - ln P(a|x), matching eqs. (3.5)-(3.6).
        The orthogonality property (eq 3.7) ensures that errors in
        estimating P(a|x) do not affect g to first order.

        Args:
            actions, states, next_actions, next_states: Observed transition
                tuples (a_t, x_t, a_{t+1}, x_{t+1}).
            feature_matrix: Utility features z(s,a,k) of shape (S, A, K).
            ccps: Estimated choice probabilities P(a|s) of shape (S, A).
            num_states, num_actions: Problem dimensions.
            gamma: Discount factor beta from the paper.

        Returns:
            Tuple of (h_table, g_table) where:
            - h_table: shape (num_states, num_actions, num_features), the
              estimated h^{(j)}(a,x) for each feature j.
            - g_table: shape (num_states, num_actions), the estimated g(a,x).
        """
        n_samples = len(states)
        num_features = feature_matrix.shape[2]

        # Build basis functions phi(a,x) and phi(a',x')
        phi = self._build_basis_functions(
            actions, states, num_states, num_actions, problem, feature_matrix
        )
        phi_next = self._build_basis_functions(
            next_actions, next_states, num_states, num_actions, problem, feature_matrix
        )
        n_basis = phi.shape[1]

        # -----------------------------------------------------------------
        # Compute the shared matrix A (eq 3.5 denominator):
        #   A = E_n[phi(a,x) * (phi(a,x) - beta * phi(a',x'))^T]
        #     = (1/N) * sum_i phi_i * (phi_i - beta * phi'_i)^T
        #
        # Lemma 2 in the paper guarantees A is non-singular when beta < 1
        # and E[phi * phi^T] is non-singular (basis functions are linearly
        # independent).
        # -----------------------------------------------------------------
        diff = phi - gamma * phi_next  # (N, n_basis)
        A = (phi.T @ diff) / n_samples  # (n_basis, n_basis)

        # Regularize slightly for numerical stability. High-dimensional
        # encoded bases can be nearly collinear in finite samples, so expose
        # this as a conservative tuning knob.
        A += self._config.basis_ridge * np.eye(n_basis)

        # Solve A * omega = b for each component, reusing the factorization.
        # The pseudoinverse option is useful when finite-sample coverage leaves
        # weak directions in a rich encoded basis.
        if self._config.basis_pinv_rcond is not None and self._config.basis_pinv_rcond > 0:
            A_inv = np.linalg.pinv(A, rcond=self._config.basis_pinv_rcond)
        else:
            A_inv = np.linalg.solve(A, np.eye(n_basis))

        # -----------------------------------------------------------------
        # Solve for h^{(j)}(a,x) = phi(a,x)^T * omega^{(j)} for each j.
        #
        # For component j: b^{(j)} = E_n[phi(a,x) * z_j(a_t, x_t)]
        # where z_j(a,x) = feature_matrix[x, a, j].
        # -----------------------------------------------------------------
        h_omega = np.zeros((n_basis, num_features), dtype=np.float64)
        for j in range(num_features):
            # z_j values for each observed (a,x) pair
            z_j = np.array([feature_matrix[s, a, j] for s, a in zip(states, actions)])
            b_j = (phi.T @ z_j) / n_samples  # (n_basis,)
            h_omega[:, j] = A_inv @ b_j

        # -----------------------------------------------------------------
        # Solve for g(a,x) = r(a,x)^T * xi.
        #
        # g satisfies: g(a,x) = beta * E[e(a',x') + g(a',x') | a,x]
        # where e(a,x) = euler_constant - ln P(a|x).
        #
        # The Euler-Mascheroni constant is approximately 0.5772.
        # e(a,x) captures the expected value of the Type I EV error term
        # conditional on choosing action a (Hotz-Miller inversion).
        #
        # We use the same basis functions for g as for h. The paper allows
        # different basis functions r(a,x) but using the same phi works.
        # -----------------------------------------------------------------
        EULER_MASCHERONI = 0.5772156649015329

        safe_ccps = np.clip(np.array(ccps), 1e-10, 1.0)
        # beta * e(a',x') for each observed transition tuple.
        e_vals = gamma * np.array([
            EULER_MASCHERONI - np.log(safe_ccps[s, a])
            for s, a in zip(next_states, next_actions)
        ])
        b_g = (phi.T @ e_vals) / n_samples
        g_omega = A_inv @ b_g

        # -----------------------------------------------------------------
        # Evaluate h and g on all (state, action) pairs to build tables.
        # This gives us tabular h(a,x) and g(a,x) for use in the
        # pseudo-log-likelihood.
        # -----------------------------------------------------------------
        h_table = np.zeros((num_states, num_actions, num_features), dtype=np.float64)
        g_table = np.zeros((num_states, num_actions), dtype=np.float64)

        for a in range(num_actions):
            all_s = np.arange(num_states)
            all_a = np.full(num_states, a, dtype=np.int32)
            phi_sa = self._build_basis_functions(
                all_a, all_s, num_states, num_actions, problem, feature_matrix
            )
            for j in range(num_features):
                h_table[:, a, j] = phi_sa @ h_omega[:, j]
            g_table[:, a] = phi_sa @ g_omega

        return h_table, g_table

    # ==================================================================
    # Step 3b: Neural AVI method (Section 3.2, Algorithm 1)
    # ==================================================================
    # The paper's AVI algorithm iteratively refines h(a,x) by solving
    # a regression problem at each iteration:
    #
    #   h_{j+1} = argmin_f E_n[(z(a,x) + beta * h_j(a',x') - f(a,x))^2]
    #
    # The target is z(a,x) + beta * h_j(a',x') where (a,x,a',x') are
    # observed transitions. The key point: h is a function of (action,
    # state), not just state. We use the actual next (a',x') pair.
    #
    # For g: replace z(a,x) with beta * e(a',x') where
    # e(a,x) = euler - ln P(a|x).
    #
    # The paper's initialization (practical recommendations):
    #   h_1(a,x) = (1-beta)^{-1} * E_n[z(a,x)]
    #   g_1(a,x) = beta * (1-beta)^{-1} * E_n[e(a',x')]
    # ==================================================================

    def _build_state_features(self, states: jnp.ndarray, problem: DDCProblem) -> jnp.ndarray:
        """Create neural network input features from state indices.

        Uses the problem's state_encoder if available, otherwise normalizes
        state indices to [0, 1]. The encoder output is concatenated with
        a one-hot action encoding when learning h(a,x) directly.
        """
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        denom = max(problem.num_states - 1, 1)
        return (states.astype(jnp.float64) / denom)[:, None]

    def _build_action_state_features(
        self,
        actions: np.ndarray,
        states: np.ndarray,
        problem: DDCProblem,
    ) -> jnp.ndarray:
        """Build features for (action, state) pairs for neural h(a,x).

        Concatenates state features with a one-hot action encoding.
        The paper's h(a,x) is a function of both action and state.

        Returns:
            Feature array of shape (n_samples, state_dim + num_actions).
        """
        state_feats = self._build_state_features(jnp.array(states), problem)
        # One-hot encode actions
        action_onehot = jnp.eye(problem.num_actions)[jnp.array(actions)]
        return jnp.concatenate([state_feats, action_onehot], axis=1)

    def _neural_avi_solve(
        self,
        actions: np.ndarray,
        states: np.ndarray,
        next_actions: np.ndarray,
        next_states: np.ndarray,
        feature_matrix: np.ndarray,
        ccps: np.ndarray,
        problem: DDCProblem,
        gamma: float,
        key: jax.Array,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Estimate h(a,x) and g(a,x) via neural AVI (Algorithm 1).

        Implements the paper's AVI algorithm with neural networks. For each
        component j of h, a separate network is trained. An additional
        network is trained for g.

        Each AVI iteration:
        1. Freeze current network (stop_gradient on target).
        2. Compute targets: z_j(a,x) + beta * h_j^{curr}(a',x').
        3. Train network to minimize MSE between predictions and targets.

        Initialization follows the paper's recommendation:
            h_1 = (1-beta)^{-1} * E_n[z(a,x)]
            g_1 = beta * (1-beta)^{-1} * E_n[e(a',x')]

        Early stopping monitors relative change (footnote 9 of the paper):
            epsilon_j = E_n[||h_{j+1} - h_j||^2] / E_n[||h_j - E[h_j]||^2]
        and stops when epsilon_j < avi_early_stop_tol.

        Returns:
            Tuple of (h_table, g_table, loss_histories).
        """
        cfg = self._config
        num_features = feature_matrix.shape[2]
        num_states = problem.num_states
        num_actions = problem.num_actions
        EULER_MASCHERONI = 0.5772156649015329

        # Build (action, state) features for neural network input
        # The paper's h(a,x) takes both action and state as input
        feat_ax = self._build_action_state_features(actions, states, problem)
        feat_ax_next = self._build_action_state_features(next_actions, next_states, problem)
        input_dim = feat_ax.shape[1]

        n_samples = len(states)
        loss_histories = {}

        # -----------------------------------------------------------------
        # Compute z_j(a,x) values for each transition tuple.
        # z(a,x) = feature_matrix[x, a, :] is the known utility feature vector.
        # -----------------------------------------------------------------
        z_values = np.array([
            feature_matrix[s, a]
            for s, a in zip(states, actions)
        ])  # (N, K)

        # Compute e(a,x) = euler - ln P(a|x) for g. The g recursion in
        # eq. (2.2) is beta * E[e(a',x') + g(a',x') | a,x], so the AVI
        # target uses next-period e values.
        safe_ccps = np.clip(np.array(ccps), 1e-10, 1.0)
        e_next = np.array([
            EULER_MASCHERONI - np.log(safe_ccps[s, a])
            for s, a in zip(next_states, next_actions)
        ])  # (N,)

        # -----------------------------------------------------------------
        # Train one network per component of h, plus one for g.
        # -----------------------------------------------------------------
        h_nets = []
        for j in range(num_features):
            self._log(f"Training h component {j} via neural AVI")
            key, net_key = jax.random.split(key)

            # Paper initialization: h_1 = (1-beta)^{-1} * E_n[z_j(a,x)]
            init_value = float(np.mean(z_values[:, j])) / (1.0 - gamma)

            net, losses = self._train_single_avi_network(
                feat_ax=feat_ax,
                feat_ax_next=feat_ax_next,
                reward_values=jnp.array(z_values[:, j]),
                init_value=init_value,
                gamma=gamma,
                key=net_key,
                input_dim=input_dim,
            )
            h_nets.append(net)
            loss_histories[f"h_{j}"] = losses

        # Train g network
        self._log("Training g via neural AVI")
        key, g_key = jax.random.split(key)
        # Paper initialization: g_1 = beta * (1-beta)^{-1} * E_n[e(a',x')]
        g_init = gamma * float(np.mean(e_next)) / (1.0 - gamma)

        g_net, g_losses = self._train_single_avi_network(
            feat_ax=feat_ax,
            feat_ax_next=feat_ax_next,
            reward_values=jnp.array(gamma * e_next),
            init_value=g_init,
            gamma=gamma,
            key=g_key,
            input_dim=input_dim,
        )
        loss_histories["g"] = g_losses

        # -----------------------------------------------------------------
        # Evaluate h and g on all (state, action) pairs to build tables
        # -----------------------------------------------------------------
        h_table = np.zeros((num_states, num_actions, num_features), dtype=np.float64)
        g_table = np.zeros((num_states, num_actions), dtype=np.float64)

        for a in range(num_actions):
            all_s = np.arange(num_states)
            all_a = np.full(num_states, a, dtype=np.int32)
            feat_sa = self._build_action_state_features(all_a, all_s, problem)
            for j in range(num_features):
                h_table[:, a, j] = np.array(jax.vmap(h_nets[j])(feat_sa))
            g_table[:, a] = np.array(jax.vmap(g_net)(feat_sa))

        return h_table, g_table, loss_histories

    def _train_single_avi_network(
        self,
        feat_ax: jnp.ndarray,
        feat_ax_next: jnp.ndarray,
        reward_values: jnp.ndarray,
        init_value: float,
        gamma: float,
        key: jax.Array,
        input_dim: int,
    ) -> tuple[_EVComponentNetwork, list[float]]:
        """Train a single component network via AVI (Algorithm 1).

        AVI iteration j:
            target_i = reward_i + beta * net_frozen(feat_next_i)
            net_{j+1} = argmin_f (1/N) sum_i (f(feat_i) - target_i)^2

        The paper notes that each iteration is a standard prediction/
        regression problem. The "reward" is z_j(a,x) for h components
        or e(a,x) for g.

        Early stopping follows footnote 9:
            epsilon_j = mean(||h_{j+1} - h_j||^2) / var(h_j)
            Stop when epsilon_j < avi_early_stop_tol.
        """
        cfg = self._config

        key, init_key = jax.random.split(key)
        net = _EVComponentNetwork(
            input_dim,
            cfg.hidden_dim,
            cfg.num_hidden_layers,
            key=init_key,
            output_shift=init_value,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(cfg.learning_rate),
        )
        opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

        n_samples = len(feat_ax)
        losses = []

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

        prev_preds = None
        for avi_iter in pbar:
            # Compute TD targets with frozen network (semi-gradient)
            # Target = reward(a,x) + beta * h_current(a', x')
            v_next = jax.lax.stop_gradient(jax.vmap(net)(feat_ax_next))
            targets = reward_values + gamma * v_next

            # Record predictions before training for early stopping check
            if cfg.avi_early_stop_tol > 0 and avi_iter > 0:
                prev_preds = jax.vmap(net)(feat_ax)

            # Train for epochs_per_avi epochs on these frozen targets
            for epoch in range(cfg.epochs_per_avi):
                key, perm_key = jax.random.split(key)
                perm = jax.random.permutation(perm_key, n_samples)

                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_samples, cfg.batch_size):
                    end = min(start + cfg.batch_size, n_samples)
                    idx = perm[start:end]
                    net, opt_state, loss = train_step(
                        net, opt_state, feat_ax[idx], targets[idx]
                    )
                    epoch_loss += float(loss)
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "avi": f"{avi_iter+1}/{cfg.avi_iterations}",
                "ep/avi": cfg.epochs_per_avi,
            })

            # ---------------------------------------------------------
            # Early stopping check (footnote 9 of the paper):
            #   epsilon_j = E[||h_{j+1} - h_j||^2] / Var(h_j)
            #   Stop when epsilon_j < avi_early_stop_tol
            #
            # This measures whether the AVI updates have converged:
            # when changes are small relative to the signal, stop.
            # ---------------------------------------------------------
            if cfg.avi_early_stop_tol > 0 and prev_preds is not None:
                curr_preds = jax.vmap(net)(feat_ax)
                change_sq = float(jnp.mean((curr_preds - prev_preds) ** 2))
                variance = float(jnp.var(prev_preds))
                if variance > 1e-12:
                    epsilon_j = change_sq / variance
                    if epsilon_j < cfg.avi_early_stop_tol:
                        self._log(f"  AVI early stop at iter {avi_iter + 1}, epsilon={epsilon_j:.6f}")
                        break

        return net, losses

    # ==================================================================
    # Step 4: Pseudo-log-likelihood and partial MLE
    # ==================================================================
    # The pseudo-log-likelihood from equation (2.1) is:
    #
    #   Q(theta) = E_n[ln pi(a,x; theta, h, g)]
    #
    # where pi(a,x; theta, h, g) = exp(h(a,x)^T theta + g(a,x))
    #                              / sum_a' exp(h(a',x)^T theta + g(a',x))
    #
    # This is a standard softmax/multinomial logit, where the choice value
    # of action a at state x is:
    #   v(a,x) = h(a,x)^T theta + g(a,x)
    #
    # h(a,x)^T theta already includes the current flow feature z(a,x)^T theta
    # by eq. (2.2); adding z(a,x)^T theta again double-counts flow utility.
    # ==================================================================

    def _pseudo_log_likelihood_jax(
        self,
        params: jnp.ndarray,
        h_table: jnp.ndarray,
        g_table: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        obs_states: jnp.ndarray,
        obs_actions: jnp.ndarray,
        sigma: float,
    ) -> jnp.ndarray:
        """Compute the pseudo-log-likelihood Q(theta) from equation (2.1).

        Pure JAX implementation so jax.grad can differentiate through it.

        v(a,x) = h(a,x)^T * theta + g(a,x)

        Q(theta) = (1/N) * sum_i ln softmax(v(a_i, x_i) / sigma)

        where the softmax is over all actions a' at state x_i.
        """
        h_weighted = jnp.einsum("sak,k->sa", h_table, params)
        v = h_weighted + g_table

        log_probs = jax.nn.log_softmax(v / sigma, axis=1)
        return log_probs[obs_states, obs_actions].sum()

    def _partial_mle(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        h_table: np.ndarray,
        g_table: np.ndarray,
        initial_params: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, int, int, str, bool]:
        """Optimize structural parameters theta via partial MLE.

        Maximizes Q(theta) = E_n[ln pi(a,x; theta, h, g)] from eq (2.1)
        using L-BFGS-B, where h and g are treated as fixed (estimated
        in the first stage).

        This is the "plug-in" estimator theta_tilde from the paper. For
        valid inference, the locally robust correction (Section 4) is
        applied separately.
        """
        sigma = problem.scale_parameter
        feature_matrix_jax = jnp.array(utility.feature_matrix, dtype=jnp.float64)
        h_table_jax = jnp.array(h_table, dtype=jnp.float64)
        g_table_jax = jnp.array(g_table, dtype=jnp.float64)
        obs_states_jax = jnp.array(panel.get_all_states())
        obs_actions_jax = jnp.array(panel.get_all_actions())

        n_obs = max(int(obs_states_jax.shape[0]), 1)

        def neg_ll(params):
            # The paper's criterion Q(theta) is an empirical average. Keep
            # the optimizer on that scale so outer_tol is sample-size stable;
            # report the summed log-likelihood below for standard model stats.
            loss = -self._pseudo_log_likelihood_jax(
                params, h_table_jax, g_table_jax, feature_matrix_jax,
                obs_states_jax, obs_actions_jax, sigma,
            ) / n_obs
            if self._config.theta_l2_penalty > 0:
                loss = (
                    loss
                    + 0.5
                    * self._config.theta_l2_penalty
                    * jnp.sum(params**2)
                    / n_obs
                )
            return loss

        if initial_params is None:
            initial_params = np.array(utility.get_initial_parameters())

        lower, upper = utility.get_parameter_bounds()

        result = minimize_lbfgsb(
            neg_ll,
            jnp.asarray(initial_params, dtype=jnp.float64),
            bounds=(jnp.asarray(lower, dtype=jnp.float64), jnp.asarray(upper, dtype=jnp.float64)),
            maxiter=self._config.outer_max_iter,
            tol=self._config.outer_tol,
            desc="TD-CCP partial MLE",
        )

        params_opt = np.array(result.x)
        ll_opt = float(self._pseudo_log_likelihood_jax(
            jnp.asarray(params_opt, dtype=jnp.float64),
            h_table_jax,
            g_table_jax,
            feature_matrix_jax,
            obs_states_jax,
            obs_actions_jax,
            sigma,
        ))
        return params_opt, ll_opt, result.nit, result.nfev, str(result.message), bool(result.success)

    # ==================================================================
    # Step 5: Locally robust inference (Section 4)
    # ==================================================================
    # The naive plug-in estimator theta_tilde has standard errors that
    # depend on the first-stage estimation error of h and g. The paper
    # constructs a locally robust moment zeta (eq 4.6) that adds a
    # correction involving a "backward" value function lambda.
    #
    # The key idea: the moment condition for theta_tilde is
    #   m(a,x; theta, h, g) = d/d_theta ln pi(a,x; theta, h, g)
    #
    # The locally robust moment adds a correction:
    #   zeta(a,x; theta, h, g, lambda) = m(a,x; theta, h, g) + correction
    #
    # where the correction involves lambda, a backward projection that
    # regresses current states onto future states. This correction makes
    # the moment insensitive to first-order perturbations in h and g,
    # achieving sqrt(n) rates despite nonparametric first stages.
    #
    # The sandwich standard errors are then:
    #   V = (G^T Omega^{-1} G)^{-1}
    #   G = E_n[d/d_theta zeta]
    #   Omega = E_n[zeta * zeta^T]
    # ==================================================================

    def _compute_backward_value(
        self,
        params: np.ndarray,
        h_table: np.ndarray,
        g_table: np.ndarray,
        feature_matrix: np.ndarray,
        actions: np.ndarray,
        states: np.ndarray,
        next_actions: np.ndarray,
        next_states: np.ndarray,
        problem: DDCProblem,
        gamma: float,
    ) -> np.ndarray:
        """Estimate the backward value function lambda (Algorithm 5).

        Lambda is defined implicitly through a backward Bellman equation.
        For each parameter component j, lambda_j(a,x) satisfies:

            lambda_j(a,x) = d/d_h_j m(a,x; theta, h, g)
                          + beta * E[lambda_j(a',x') | a,x]

        where m is the score of the pseudo-log-likelihood. In practice,
        we approximate lambda using the same semi-gradient or AVI approach
        but with the backward TD operator.

        For the logit model, the score contribution from h_j is:
            d/d_h_j m = theta_j * (indicator(a_obs) - pi(a|x))

        We use linear semi-gradient to estimate lambda for simplicity.

        Returns:
            lambda_table of shape (num_states, num_actions, num_params).
        """
        num_states = problem.num_states
        num_actions = problem.num_actions
        num_params = len(params)
        sigma = problem.scale_parameter

        # Compute policy pi(a|x) from current parameters
        h_weighted = np.einsum("sak,k->sa", h_table, params)
        v = (h_weighted + g_table) / sigma
        v_max = v.max(axis=1, keepdims=True)
        exp_v = np.exp(v - v_max)
        pi = exp_v / exp_v.sum(axis=1, keepdims=True)

        # Score of the pseudo-log-likelihood with respect to h_j:
        # For observation (a_i, x_i):
        #   dm/dh_j = theta_j * (1{a=a_i} - pi(a_i|x_i)) / sigma
        #
        # This is the "reward" for the backward Bellman equation defining lambda.
        # We solve the backward equation using the semi-gradient method.

        # Build basis functions for the backward solve
        phi = self._build_basis_functions(
            actions, states, num_states, num_actions, problem, feature_matrix
        )
        phi_next = self._build_basis_functions(
            next_actions, next_states, num_states, num_actions, problem, feature_matrix
        )
        n_basis = phi.shape[1]
        n_samples = len(states)

        # The backward semi-gradient matrix uses REVERSED time direction:
        # A_back = E_n[phi(a',x') * (phi(a',x') - beta * phi(a,x))^T]
        # This regresses future onto past instead of past onto future.
        diff_back = phi_next - gamma * phi  # (N, n_basis)
        A_back = (phi_next.T @ diff_back) / n_samples
        A_back += self._config.basis_ridge * np.eye(n_basis)
        if self._config.basis_pinv_rcond is not None and self._config.basis_pinv_rcond > 0:
            A_back_inv = np.linalg.pinv(A_back, rcond=self._config.basis_pinv_rcond)
        else:
            A_back_inv = np.linalg.solve(A_back, np.eye(n_basis))

        lambda_table = np.zeros((num_states, num_actions, num_params), dtype=np.float64)

        for j in range(num_params):
            # "reward" for backward equation: dm/dh_j at observed (a,x)
            reward_j = np.array([
                params[j] * (1.0 - pi[s, a]) / sigma
                for s, a in zip(states, actions)
            ])
            b_back = (phi_next.T @ reward_j) / n_samples
            lambda_omega_j = A_back_inv @ b_back

            # Evaluate on all (state, action) pairs
            for a in range(num_actions):
                all_s = np.arange(num_states)
                all_a = np.full(num_states, a, dtype=np.int32)
                phi_sa = self._build_basis_functions(
                    all_a, all_s, num_states, num_actions, problem, feature_matrix
                )
                lambda_table[:, a, j] = phi_sa @ lambda_omega_j

        return lambda_table

    def _compute_robust_se(
        self,
        params: np.ndarray,
        h_table: np.ndarray,
        g_table: np.ndarray,
        lambda_table: np.ndarray,
        feature_matrix: np.ndarray,
        actions: np.ndarray,
        states: np.ndarray,
        next_actions: np.ndarray,
        next_states: np.ndarray,
        ccps: np.ndarray,
        problem: DDCProblem,
        gamma: float,
        individual_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute locally robust standard errors (Section 4.2.1).

        The locally robust moment is (eq 4.6):
            zeta_i = m_i + correction_i

        where m_i is the score of the pseudo-log-likelihood at observation i,
        and the correction accounts for first-stage estimation error in h and g.

        The correction involves the backward value function lambda and the
        TD errors of h and g at each observation.

        Standard errors are computed via the sandwich formula:
            V_hat = (G^T Omega^{-1} G)^{-1}
            G = (1/N) sum_i d/d_theta zeta_i
            Omega = (1/N) sum_i zeta_i * zeta_i^T

        Returns:
            Hessian-like matrix for SE computation, shape (K, K).
        """
        num_params = len(params)
        sigma = problem.scale_parameter
        n_samples = len(states)
        EULER_MASCHERONI = 0.5772156649015329

        # Compute policy pi(a|x)
        h_weighted = np.einsum("sak,k->sa", h_table, params)
        v = (h_weighted + g_table) / sigma
        v_max = v.max(axis=1, keepdims=True)
        exp_v = np.exp(v - v_max)
        pi = exp_v / exp_v.sum(axis=1, keepdims=True)

        safe_ccps = np.clip(np.array(ccps), 1e-10, 1.0)

        # -----------------------------------------------------------------
        # Compute zeta_i for each observation i.
        #
        # The score m_i = d/d_theta ln pi(a_i | x_i; theta, h, g):
        #   m_i = (h(a_i,x_i) - sum_a' pi(a'|x_i)*h(a',x_i)) / sigma
        #
        # The correction involves:
        # - TD error of h: delta_h_j = z_j(a,x) + beta*h_j(a',x') - h_j(a,x)
        # - TD error of g: delta_g = beta*e(a',x') + beta*g(a',x') - g(a,x)
        # - lambda_j(a',x'): backward value function at next state
        #
        # zeta_i = m_i + sum_j lambda_j(a'_i, x'_i) * delta_h_j_i * theta_j
        #        + lambda_g_correction
        # -----------------------------------------------------------------

        zeta = np.zeros((n_samples, num_params), dtype=np.float64)

        for i in range(n_samples):
            s, a = states[i], actions[i]
            s_next, a_next = next_states[i], next_actions[i]

            # Score: m_i
            z_obs = h_table[s, a]  # (K,)
            z_expected = np.sum(pi[s, :, None] * h_table[s], axis=0)
            m_i = (z_obs - z_expected) / sigma  # (K,)

            # TD errors for h components
            for j in range(num_params):
                delta_h_j = (
                    feature_matrix[s, a, j]
                    + gamma * h_table[s_next, a_next, j]
                    - h_table[s, a, j]
                )
                # Correction: lambda_j at CURRENT state (a,x) per paper eq (4.4)
                m_i[j] += lambda_table[s, a, j] * delta_h_j

            # TD error for g
            e_next = EULER_MASCHERONI - np.log(safe_ccps[s_next, a_next])
            delta_g = gamma * e_next + gamma * g_table[s_next, a_next] - g_table[s, a]
            lambda_g = np.mean(lambda_table[s, a])
            m_i += lambda_g * delta_g * np.ones(num_params) / sigma

            zeta[i] = m_i

        # -----------------------------------------------------------------
        # Sandwich formula: V = (G^T Omega^{-1} G)^{-1}
        # We cluster Omega by individual when individual_ids is provided.
        # Without clustering, panel data with T > 1 periods per individual
        # inflates n_samples by (T-1), understating variance by sqrt(T-1).
        # -----------------------------------------------------------------
        if individual_ids is not None and len(individual_ids) == n_samples:
            # Cluster: sum zeta within individual, then outer product
            n_ind = int(individual_ids.max()) + 1
            zeta_clustered = np.zeros((n_ind, num_params), dtype=np.float64)
            for i in range(n_samples):
                zeta_clustered[individual_ids[i]] += zeta[i]
            Omega = (zeta_clustered.T @ zeta_clustered) / n_ind  # (K, K)
            n_eff = n_ind  # effective sample size for SE scaling
        else:
            Omega = (zeta.T @ zeta) / n_samples  # (K, K)
            n_eff = n_samples

        # Numerical Jacobian G = d/d_theta E_n[zeta]
        eps = 1e-5
        G = np.zeros((num_params, num_params), dtype=np.float64)
        zeta_mean = zeta.mean(axis=0)  # (K,)

        # For the score-based moment, G is approximately the negative Hessian
        # divided by n_samples. We approximate it numerically.
        for j in range(num_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += eps
            params_minus[j] -= eps

            # Recompute score at perturbed params.
            h_w_plus = np.einsum("sak,k->sa", h_table, params_plus)
            v_plus = (h_w_plus + g_table) / sigma
            v_max_p = v_plus.max(axis=1, keepdims=True)
            pi_plus = np.exp(v_plus - v_max_p) / np.exp(v_plus - v_max_p).sum(axis=1, keepdims=True)

            h_w_minus = np.einsum("sak,k->sa", h_table, params_minus)
            v_minus = (h_w_minus + g_table) / sigma
            v_max_m = v_minus.max(axis=1, keepdims=True)
            pi_minus = np.exp(v_minus - v_max_m) / np.exp(v_minus - v_max_m).sum(axis=1, keepdims=True)

            # Mean score at plus and minus
            score_plus = np.zeros(num_params)
            score_minus = np.zeros(num_params)
            for i in range(n_samples):
                s, a = states[i], actions[i]
                z_obs = h_table[s, a]
                z_exp_p = np.sum(pi_plus[s, :, None] * h_table[s], axis=0)
                z_exp_m = np.sum(pi_minus[s, :, None] * h_table[s], axis=0)
                score_plus += (z_obs - z_exp_p) / sigma
                score_minus += (z_obs - z_exp_m) / sigma

            G[:, j] = (score_plus - score_minus) / (2 * eps * n_samples)

        # V = (G^T Omega^{-1} G)^{-1}
        # SE = sqrt(diag(V / n_eff)).  Base class computes sqrt(diag(inv(-H))),
        # so we need -H = n_eff * GtOiG, i.e. H = -n_eff * GtOiG.
        try:
            Omega_inv = np.linalg.solve(Omega + 1e-10 * np.eye(num_params), np.eye(num_params))
            GtOiG = G.T @ Omega_inv @ G
            hessian_equiv = -n_eff * GtOiG  # base class inverts (-H) to get V/n_eff
        except np.linalg.LinAlgError:
            self._log("Warning: robust SE computation failed, falling back to naive Hessian")
            hessian_equiv = None

        return hessian_equiv

    # ==================================================================
    # Clustered SE computation
    # ==================================================================

    def _compute_clustered_se(
        self,
        params: np.ndarray,
        h_table: np.ndarray,
        g_table: np.ndarray,
        feature_matrix: np.ndarray,
        all_states: np.ndarray,
        all_actions: np.ndarray,
        panel: "Panel",
        sigma: float,
        H: np.ndarray,
    ) -> jnp.ndarray | None:
        """Clustered MLE sandwich SE.

        Var(theta_hat) = H^{-1} B_cluster H^{-1}

        where H is the Hessian of the total pseudo-log-likelihood and
        B_cluster = Σ_g (Σ_{i in g} g_i)(Σ_{i in g} g_i)^T with g_i the
        per-observation pseudo-LL score.  Summing scores within individual g
        before taking the outer product clusters standard errors at the
        individual level, which is correct when the panel has T > 1 periods
        per person.

        all_states / all_actions should be consistent with H (same N=T×n_ind).
        We build per-obs individual IDs directly from panel.trajectories.

        Returns the effective hessian H_eff = -inv(V) so the base class
        formula SE = sqrt(diag(inv(-H_eff))) recovers SE = sqrt(diag(V)).
        """
        n_samples = len(all_states)
        n_params = len(params)

        # Build per-obs individual IDs (length = T × n_ind = n_samples)
        obs_ind_ids = []
        for ind_idx, traj in enumerate(panel.trajectories):
            n_obs = len(traj.states)
            if n_obs > 0:
                obs_ind_ids.append(np.full(n_obs, ind_idx, dtype=np.int32))
        if not obs_ind_ids:
            return None
        obs_ind_ids = np.concatenate(obs_ind_ids)  # length = n_samples

        if len(obs_ind_ids) != n_samples:
            # Mismatch — fall back gracefully
            return None

        # Per-observation scores: d/dtheta log pi(a_i | s_i; theta, h, g)
        z_h = h_table  # (S, A, K); h already includes current flow features
        v = np.einsum("sak,k->sa", z_h, params) + g_table  # (S, A)
        v_max = v.max(axis=1, keepdims=True)
        exp_v = np.exp(v - v_max)
        pi = exp_v / exp_v.sum(axis=1, keepdims=True)

        score_i = np.zeros((n_samples, n_params), dtype=np.float64)
        for i in range(n_samples):
            s, a = int(all_states[i]), int(all_actions[i])
            z_obs = z_h[s, a]                                    # (K,)
            z_exp = np.einsum("a,ak->k", pi[s], z_h[s])         # (K,)
            score_i[i] = (z_obs - z_exp) / sigma

        # Cluster: sum scores within individual
        n_ind = int(obs_ind_ids.max()) + 1
        cluster_scores = np.zeros((n_ind, n_params), dtype=np.float64)
        for i in range(n_samples):
            cluster_scores[obs_ind_ids[i]] += score_i[i]

        # Meat: B_cluster = Σ_g g_g g_g^T
        B_cluster = cluster_scores.T @ cluster_scores

        # Small-sample correction (Stata-style)
        G = n_ind
        N = n_samples
        K = n_params
        correction = (G / (G - 1)) * (N - 1) / (N - K) if G > 1 and N > K else 1.0

        # Sandwich: V = H_inv @ B_cluster @ H_inv
        try:
            H_inv = np.linalg.inv(-H)   # -H is positive semi-definite at optimum
            V = correction * (H_inv @ B_cluster @ H_inv)
            H_eff = -np.linalg.inv(V)   # base class computes inv(-H_eff) = V → SE = sqrt(diag(V))
            return jnp.array(H_eff)
        except np.linalg.LinAlgError:
            self._log("Warning: clustered SE computation failed, falling back to naive Hessian")
            return None

    # ==================================================================
    # Main estimation routine
    # ==================================================================

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run TD-CCP estimation.

        The full algorithm follows Adusumilli and Eckardt (2025):

        1. Estimate CCPs from data (frequency or logit).
        2. Extract observed (a,x,a',x') transition tuples.
        3. Estimate h(a,x) and g(a,x) via linear semi-gradient or neural AVI.
        4. Maximize pseudo-log-likelihood Q(theta) via L-BFGS-B.
        5. (Optional) If cross_fitting=True, repeat steps 3-4 on each fold
           and average the two theta estimates (Algorithm 2).
        6. (Optional) Compute locally robust standard errors (Section 4).
        7. (Optional) If n_policy_iterations > 1, re-solve for the policy
           and iterate (NPL refinement, not in the paper).
        """
        start_time = time.time()
        cfg = self._config
        num_states = problem.num_states
        num_actions = problem.num_actions
        gamma = problem.discount_factor
        feature_matrix = np.array(utility.feature_matrix)

        key = jax.random.PRNGKey(self._seed)

        # -----------------------------------------------------------------
        # Step 1: Estimate CCPs from data
        # -----------------------------------------------------------------
        self._log("Step 1: Estimating CCPs from data")
        ccps = self._estimate_ccps(panel, num_states, num_actions)

        # -----------------------------------------------------------------
        # Step 2: Extract observed transition tuples (a, x, a', x')
        # -----------------------------------------------------------------
        self._log("Step 2: Extracting transition tuples")
        actions, states, next_actions, next_states = self._extract_transitions(panel)
        individual_ids = self._extract_individual_ids(panel)
        n_transitions = len(states)
        self._log(f"  {n_transitions} transition tuples extracted")

        # -----------------------------------------------------------------
        # Steps 3-5: Estimate h, g and optimize theta
        # With cross-fitting (Algorithm 2) if enabled.
        # -----------------------------------------------------------------
        if cfg.cross_fitting:
            params_opt, ll_opt, total_nit, total_nfev, opt_msg, opt_success, h_table, g_table, loss_hists = \
                self._estimate_with_cross_fitting(
                    panel, utility, problem, transitions, ccps,
                    actions, states, next_actions, next_states,
                    feature_matrix, gamma, key, initial_params,
                )
        else:
            h_table, g_table, loss_hists = self._estimate_h_g(
                actions, states, next_actions, next_states,
                feature_matrix, np.array(ccps), problem, gamma, key,
            )
            params_opt, ll_opt, total_nit, total_nfev, opt_msg, opt_success = self._partial_mle(
                panel, utility, problem, h_table, g_table, initial_params,
            )

        self._log(f"  Params: {params_opt}, LL: {ll_opt:.4f}")

        # -----------------------------------------------------------------
        # Optional NPL refinement (not in paper, for n_policy_iterations > 1)
        # -----------------------------------------------------------------
        for pi_iter in range(1, cfg.n_policy_iterations):
            self._log(f"NPL iteration {pi_iter + 1}/{cfg.n_policy_iterations}")

            # Re-solve for policy with current params
            reward_matrix = utility.compute(jnp.array(params_opt))
            operator = SoftBellmanOperator(problem, transitions)
            vi_result = value_iteration(operator, reward_matrix, tol=1e-8, max_iter=5000)
            new_ccps = np.array(vi_result.policy)

            policy_change = float(np.abs(new_ccps - np.array(ccps)).max())
            self._log(f"  Policy change: {policy_change:.6f}")
            if policy_change < cfg.policy_iteration_tol:
                break

            ccps = jnp.array(new_ccps)

            # Re-estimate h, g with new CCPs
            key, iter_key = jax.random.split(key)
            h_table, g_table, iter_losses = self._estimate_h_g(
                actions, states, next_actions, next_states,
                feature_matrix, np.array(ccps), problem, gamma, iter_key,
            )
            params_opt, ll_opt, nit, nfev, opt_msg, iter_success = self._partial_mle(
                panel, utility, problem, h_table, g_table,
                np.array(params_opt),
            )
            opt_success = bool(opt_success and iter_success)
            total_nit += nit
            total_nfev += nfev

        # -----------------------------------------------------------------
        # Final policy and value function via exact value iteration
        # -----------------------------------------------------------------
        reward_matrix = utility.compute(jnp.array(params_opt))
        operator = SoftBellmanOperator(problem, transitions)
        vi_result = value_iteration(operator, reward_matrix, tol=1e-8, max_iter=5000)
        policy = vi_result.policy
        V = vi_result.V

        # -----------------------------------------------------------------
        # Standard errors
        # -----------------------------------------------------------------
        hessian = None
        if cfg.compute_se:
            self._last_h_table = h_table
            self._last_g_table = g_table
            all_states_np = np.array(panel.get_all_states())
            all_actions_np = np.array(panel.get_all_actions())

            def ll_fn(p):
                return jnp.array(self._pseudo_log_likelihood_jax(
                    np.array(p), h_table, g_table, feature_matrix,
                    all_states_np, all_actions_np, problem.scale_parameter,
                ))

            H = compute_numerical_hessian(jnp.array(params_opt), ll_fn)

            if cfg.robust_se and cfg.cross_fitting:
                # With correct cross-fitting (individual-level fold split), each
                # theta_k's pseudo-LL scores are independent of the h,g used in that
                # fold (Neyman orthogonality, Theorem 5). Standard MLE Hessian is valid.
                self._log("Using numerical Hessian for cross-fitted SE (Neyman orthogonality)")
                hessian = H
            elif cfg.robust_se:
                # Without cross-fitting, use clustered MLE sandwich: Var = H^{-1} B H^{-1}
                # This captures panel clustering but not first-stage h/g error.
                self._log("Computing clustered sandwich standard errors (no cross-fitting)")
                hessian = self._compute_clustered_se(
                    params_opt, h_table, g_table, feature_matrix,
                    all_states_np, all_actions_np, panel,
                    problem.scale_parameter, np.array(H),
                )

            if hessian is None:
                self._log("Using naive numerical Hessian (no clustering)")
                hessian = H

        optimization_time = time.time() - start_time

        # Store the log-likelihood function for external use
        self._log_likelihood_fn = lambda p: self._pseudo_log_likelihood_jax(
            np.array(p), h_table, g_table, feature_matrix,
            np.array(panel.get_all_states()),
            np.array(panel.get_all_actions()),
            problem.scale_parameter,
        )

        return EstimationResult(
            parameters=jnp.array(params_opt),
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            gradient_contributions=None,
            converged=bool(opt_success),
            num_iterations=total_nit,
            num_function_evals=total_nfev,
            num_inner_iterations=0,
            message=f"TD-CCP ({cfg.method}): {opt_msg}",
            optimization_time=optimization_time,
            metadata={
                "loss_histories": loss_hists,
                "method": cfg.method,
                "basis_type": cfg.basis_type,
                "basis_dim": cfg.basis_dim,
                "basis_include_rewards": cfg.basis_include_rewards,
                "basis_ridge": cfg.basis_ridge,
                "basis_pinv_rcond": cfg.basis_pinv_rcond,
                "ccp_method": cfg.ccp_method,
                "ccp_smoothing": cfg.ccp_smoothing,
                "theta_l2_penalty": cfg.theta_l2_penalty,
                "cross_fitting": cfg.cross_fitting,
                "robust_se": cfg.robust_se,
                "reward_matrix": reward_matrix,
                "h_table": h_table,
                "g_table": g_table,
                # Legacy compatibility: extract per-feature EV components
                "ev_features": self._extract_ev_features(h_table, params_opt),
                "ev_entropy": g_table.mean(axis=1),
            },
        )

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _estimate_h_g(
        self,
        actions, states, next_actions, next_states,
        feature_matrix, ccps, problem, gamma, key,
    ):
        """Dispatch to semi-gradient or neural AVI for h,g estimation."""
        cfg = self._config
        num_states = problem.num_states
        num_actions = problem.num_actions

        if cfg.method == "semigradient":
            self._log("Step 3: Linear semi-gradient solve (eq 3.5)")
            h_table, g_table = self._semigradient_solve(
                actions, states, next_actions, next_states,
                feature_matrix, ccps, num_states, num_actions, gamma, problem,
            )
            return h_table, g_table, {}
        else:
            self._log("Step 3: Neural AVI solve (Algorithm 1)")
            h_table, g_table, losses = self._neural_avi_solve(
                actions, states, next_actions, next_states,
                feature_matrix, ccps, problem, gamma, key,
            )
            return h_table, g_table, losses

    def _estimate_with_cross_fitting(
        self, panel, utility, problem, transitions, ccps,
        actions, states, next_actions, next_states,
        feature_matrix, gamma, key, initial_params,
    ):
        """2-fold cross-fitting (Algorithm 2 of the paper).

        Split INDIVIDUALS into two folds. For each fold k:
        1. Estimate h, g using fold k's transitions.
        2. Estimate theta on fold -k's panel using h, g from fold k.
        3. Average the two theta estimates.

        Splitting by individual (not by transition) ensures that all T
        transitions from individual i land in the same fold, so h,g
        estimated on fold k are strictly independent of fold -k's data.
        This is the condition required for Neyman orthogonality (Theorem 5).
        """
        from econirl.core.types import Panel as EconPanel

        split_unit = getattr(self._config, "split_unit", "individual")
        self._log(f"Using 2-fold cross-fitting (Algorithm 2, split_unit={split_unit})")

        individual_ids = self._extract_individual_ids(panel)
        n_ind = len(panel.trajectories)

        if split_unit == "individual":
            # Individual-level split: all transitions from individual i go to
            # the same fold, preserving the orthogonality condition required
            # by Adusumilli and Eckardt (2025) Theorem 5.
            half_ind = n_ind // 2
            key, perm_key = jax.random.split(key)
            ind_perm = np.array(jax.random.permutation(perm_key, n_ind))
            fold1_inds = ind_perm[:half_ind]
            fold2_inds = ind_perm[half_ind:]

            fold1_panel = EconPanel([panel.trajectories[i] for i in fold1_inds])
            fold2_panel = EconPanel([panel.trajectories[i] for i in fold2_inds])

            fold1_trans_mask = np.isin(individual_ids, fold1_inds)
            fold2_trans_mask = ~fold1_trans_mask

        elif split_unit == "row":
            # Transition-level split: each (s, a, s') row is randomly assigned
            # to a fold. Provided for ablations only; this breaks the
            # independence required for the orthogonality result, since
            # transitions from the same individual leak across folds.
            n_rows = len(individual_ids)
            half_rows = n_rows // 2
            key, perm_key = jax.random.split(key)
            row_perm = np.array(jax.random.permutation(perm_key, n_rows))
            fold1_rows = row_perm[:half_rows]
            fold2_rows = row_perm[half_rows:]

            fold1_trans_mask = np.zeros(n_rows, dtype=bool)
            fold1_trans_mask[fold1_rows] = True
            fold2_trans_mask = ~fold1_trans_mask

            # For the per-fold theta MLE we still need a panel. The cleanest
            # row-level analogue is to keep the full panel on each fold and
            # only swap which fold's (h, g) is plugged into the score.
            fold1_panel = panel
            fold2_panel = panel

        else:
            raise ValueError(
                f"split_unit must be 'individual' or 'row', got {split_unit!r}"
            )

        # -----------------------------------------------------------------
        # Fold 1: estimate h,g from fold 1 transitions
        # -----------------------------------------------------------------
        key, k1 = jax.random.split(key)
        h1, g1, losses1 = self._estimate_h_g(
            actions[fold1_trans_mask], states[fold1_trans_mask],
            next_actions[fold1_trans_mask], next_states[fold1_trans_mask],
            feature_matrix, np.array(ccps), problem, gamma, k1,
        )

        # -----------------------------------------------------------------
        # Fold 2: estimate h,g from fold 2 transitions
        # -----------------------------------------------------------------
        key, k2 = jax.random.split(key)
        h2, g2, losses2 = self._estimate_h_g(
            actions[fold2_trans_mask], states[fold2_trans_mask],
            next_actions[fold2_trans_mask], next_states[fold2_trans_mask],
            feature_matrix, np.array(ccps), problem, gamma, k2,
        )

        # -----------------------------------------------------------------
        # Cross-estimation: theta_k on fold k panel using h,g from fold -k.
        # Each theta_k's pseudo-LL scores are independent of the h,g used,
        # because h,g were estimated on the complementary fold's data.
        # -----------------------------------------------------------------
        params1, ll1, nit1, nfev1, msg1, success1 = self._partial_mle(
            fold1_panel, utility, problem, h2, g2, initial_params,
        )
        params2, ll2, nit2, nfev2, msg2, success2 = self._partial_mle(
            fold2_panel, utility, problem, h1, g1,
            np.array(params1) if initial_params is None else initial_params,
        )

        # Average the two estimates (Algorithm 2, Step 4)
        params_avg = (np.array(params1) + np.array(params2)) / 2.0
        ll_avg = (ll1 + ll2) / 2.0

        # Use average of h,g tables for final policy computation and numerical Hessian
        h_avg = (h1 + h2) / 2.0
        g_avg = (g1 + g2) / 2.0

        losses = {**{f"fold1_{k}": v for k, v in losses1.items()},
                  **{f"fold2_{k}": v for k, v in losses2.items()}}

        return (
            params_avg, ll_avg,
            nit1 + nit2, nfev1 + nfev2,
            f"cross-fitted: {msg1}; {msg2}",
            bool(success1 and success2),
            h_avg, g_avg, losses,
        )

    @staticmethod
    def _extract_ev_features(h_table, params):
        """Extract per-feature EV components for legacy compatibility.

        The old implementation stored ev_features (state-only expected
        value components). We reconstruct something similar by weighting
        h(a,x) by the uniform action distribution.

        Returns:
            Array of shape (num_states, num_features).
        """
        # Average h over actions to get a state-level summary
        return h_table.mean(axis=1)  # (S, K)
