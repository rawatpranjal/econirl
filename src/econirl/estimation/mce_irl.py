"""Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL).

This module implements Maximum Causal Entropy IRL from Ziebart's 2010 thesis,
which properly accounts for the causal structure in sequential decision-making.

The key difference from standard MaxEnt IRL is the backward pass uses soft
value iteration which respects that agents don't know future randomness
at decision time.

Algorithm (following Ziebart 2010):
    BACKWARD PASS:
    1. Initialize Q(s,a) and V(s) at terminal states
    2. Propagate backwards using soft Bellman:
       Q(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
       V(s) = softmax(Q(s,·))

    FORWARD PASS:
    3. Compute state visitation frequencies:
       D(s) = ρ₀(s) + Σ_{s',a} D(s') π(a|s') P(s|s',a)

    GRADIENT:
    4. Δθ = E_expert[∇R] - E_policy[∇R]
          = empirical_features - Σ_s D(s) ∇R(s)

References:
    Ziebart, B. D. (2010). Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy. PhD thesis, CMU.

    Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008).
        Maximum entropy inverse reinforcement learning. AAAI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import hybrid_iteration, policy_iteration, value_iteration, backward_induction
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class MCEIRLConfig:
    """Configuration for MCE IRL estimation."""

    # Optimization
    optimizer: Literal["L-BFGS-B", "BFGS", "gradient", "root"] = "L-BFGS-B"
    learning_rate: float = 0.02  # Lower than typical SGD; works well with Adam
    outer_tol: float = 1e-6
    outer_max_iter: int = 200
    gradient_clip: float = 1.0  # Max gradient norm (prevents divergence)
    use_adam: bool = True  # Use Adam optimizer (adaptive learning rate)
    adam_beta1: float = 0.9  # Adam first moment decay
    adam_beta2: float = 0.999  # Adam second moment decay
    adam_eps: float = 1e-8  # Adam numerical stability

    # Inner solver (soft value iteration)
    inner_solver: Literal["value", "hybrid", "policy"] = "hybrid"  # hybrid is faster
    inner_tol: float = 1e-8
    inner_max_iter: int = 10000  # Higher for high discount factors
    switch_tol: float = 1e-3  # For hybrid: switch to NK when error < this

    # State visitation computation
    svf_tol: float = 1e-8
    svf_max_iter: int = 1000

    # Occupancy distance convergence (from imitation library, Gleave & Toyer 2022)
    # Checks L-infinity distance between demo and policy occupancy measures
    occupancy_tol: float = 1e-3  # max|D_demo - D_policy| convergence threshold

    # Inference
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic", "hessian"] = "bootstrap"
    n_bootstrap: int = 100

    # Verbosity
    verbose: bool = False


class MCEIRLEstimator(BaseEstimator):
    """Maximum Causal Entropy IRL Estimator.

    Recovers reward function parameters from demonstrated behavior using
    the maximum causal entropy principle (Ziebart 2010).

    This differs from standard MaxEnt IRL in that it properly accounts for
    the causal structure of sequential decisions - agents act before observing
    future randomness.

    Parameters
    ----------
    config : MCEIRLConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Attributes
    ----------
    config : MCEIRLConfig
        Configuration object.

    Examples
    --------
    >>> from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    >>> from econirl.preferences.reward import LinearReward
    >>>
    >>> # Create estimator with custom config
    >>> config = MCEIRLConfig(verbose=True, n_bootstrap=200)
    >>> estimator = MCEIRLEstimator(config=config)
    >>>
    >>> # Estimate reward from demonstrations
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: MCEIRLConfig | None = None,
        **kwargs,
    ):
        # Build config from defaults + overrides
        if config is None:
            config = MCEIRLConfig(**kwargs)
        else:
            # Apply any kwargs as overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Map "hessian" to "asymptotic" for compatibility with standard_errors module
        # (they are semantically equivalent: both use inverse Hessian for variance)
        if config.compute_se:
            if config.se_method == "hessian":
                effective_se_method = "asymptotic"
            else:
                effective_se_method = config.se_method
        else:
            effective_se_method = "asymptotic"

        super().__init__(
            se_method=effective_se_method,
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "MCE IRL (Ziebart 2010)"

    def _soft_value_iteration(
        self,
        operator: SoftBellmanOperator,
        reward_matrix: jnp.ndarray,
        num_periods: int | None = None,
        V_init: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        """Run soft value iteration (backward pass).

        For infinite horizon: uses contraction/hybrid iteration.
        For finite horizon: uses backward induction (deterministic, no convergence needed).

        Parameters
        ----------
        operator : SoftBellmanOperator
            Bellman operator with problem and transitions.
        reward_matrix : jnp.ndarray
            Reward matrix R(s,a), shape (n_states, n_actions).
        num_periods : int, optional
            If set, use finite-horizon backward induction.

        Returns
        -------
        V : jnp.ndarray
            Soft value function, shape (n_states,) for infinite horizon,
            or shape (num_periods, n_states) for finite horizon.
        policy : jnp.ndarray
            Soft policy π(a|s), shape (n_states, n_actions) for infinite horizon,
            or shape (num_periods, n_states, n_actions) for finite horizon.
        converged : bool
            Whether the iteration converged (always True for finite horizon).
        """
        if num_periods is not None:
            # Finite horizon: backward induction (deterministic, no convergence needed)
            fh_result = backward_induction(operator, reward_matrix, num_periods)
            return fh_result.V, fh_result.policy, True

        if self.config.inner_solver == "policy":
            result = policy_iteration(
                operator,
                reward_matrix,
                V_init=V_init,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
                eval_method="matrix",
            )
            return result.V, result.policy, result.converged
        elif self.config.inner_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                reward_matrix,
                V_init=V_init,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
                switch_tol=self.config.switch_tol,
            )
            return result.V, result.policy, result.converged
        else:
            # Pure value iteration (original implementation)
            result = value_iteration(
                operator,
                reward_matrix,
                V_init=V_init,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )
            return result.V, result.policy, result.converged

    def _compute_state_visitation(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        initial_dist: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute expected state visitation frequencies (forward pass).

        Uses forward message passing to compute the expected number of times
        each state is visited under the policy.

        Parameters
        ----------
        policy : jnp.ndarray
            Policy π(a|s), shape (n_states, n_actions).
        transitions : jnp.ndarray
            Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
        problem : DDCProblem
            Problem specification.
        initial_dist : jnp.ndarray, optional
            Initial state distribution ρ₀. If None, uses uniform.

        Returns
        -------
        D : jnp.ndarray
            State visitation frequencies, shape (n_states,).
        """
        n_states = problem.num_states
        gamma = problem.discount_factor

        if initial_dist is None:
            D = jnp.ones(n_states, dtype=policy.dtype) / n_states
        else:
            D = jnp.array(initial_dist)

        # Compute policy-weighted transition: P_π(s'|s) = Σ_a π(a|s) P(s'|s,a)
        # transitions: (n_actions, n_states, n_states) = [a, from_s, to_s]
        # policy: (n_states, n_actions) = [s, a]
        P_pi = jnp.einsum("sa,ast->st", policy, transitions)

        # Fixed point iteration: D = ρ₀ + γ P_π^T D
        # Equivalently: D = (I - γ P_π^T)^{-1} ρ₀
        # But we use iteration for numerical stability

        rho0 = jnp.array(D)
        for i in range(self.config.svf_max_iter):
            D_new = rho0 + gamma * (P_pi.T @ D)

            delta = float(jnp.abs(D_new - D).max())
            D = D_new

            if delta < self.config.svf_tol:
                break

        # Normalize to probability distribution
        D = D / D.sum()

        return D

    def _compute_empirical_state_occupancy(
        self,
        panel: Panel,
        n_states: int,
        n_actions: int,
        discount: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute empirical state and state-action occupancy from demonstrations.

        Following Ziebart (2008) and the imitation library: count how often
        each state (and state-action pair) appears in the demonstrations.
        Infinite-horizon MCE feature matching uses discounted occupancy
        measures, so callers may pass ``discount=problem.discount_factor`` to
        weight period ``t`` by ``discount ** t`` before normalization.

        Returns
        -------
        D_demo : jnp.ndarray
            State occupancy, shape (n_states,). Sums to 1.
        D_demo_sa : jnp.ndarray
            State-action occupancy, shape (n_states, n_actions). Sums to 1.
        """
        D_s_np = np.zeros(n_states, dtype=np.float64)
        D_sa_np = np.zeros((n_states, n_actions), dtype=np.float64)
        total_weight = 0.0

        for traj in panel.trajectories:
            states_np = np.asarray(traj.states, dtype=np.int64)
            actions_np = np.asarray(traj.actions, dtype=np.int64)
            if len(states_np) == 0:
                continue
            if discount == 1.0:
                weights = np.ones(len(states_np), dtype=np.float64)
            else:
                weights = np.power(float(discount), np.arange(len(states_np)))

            np.add.at(D_s_np, states_np, weights)
            idx_flat = states_np * n_actions + actions_np
            np.add.at(D_sa_np.ravel(), idx_flat, weights)
            total_weight += float(weights.sum())

        if total_weight > 0:
            D_s_np = D_s_np / total_weight
            D_sa_np = D_sa_np / total_weight

        return jnp.array(D_s_np), jnp.array(D_sa_np)

    def _compute_empirical_features(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        n_states: int | None = None,
        n_actions: int | None = None,
        discount: float = 1.0,
    ) -> jnp.ndarray:
        """Compute empirical feature expectations using state-action occupancy.

        Uses the empirical state-action occupancy measure D_demo(s,a):
            μ_D = Σ_s Σ_a D_demo(s,a) φ(s,a)

        This is consistent with the occupancy-measure E_π computation,
        following Ziebart (2008) Eq. 6 and the imitation library.
        """
        # Get feature matrix
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (S, A, K)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (S, K)
        else:
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        if feature_matrix.ndim == 3:
            _ns, _na = feature_matrix.shape[0], feature_matrix.shape[1]
            D_s, D_sa = self._compute_empirical_state_occupancy(
                panel, n_states or _ns, n_actions or _na, discount=discount
            )
            # μ_D = Σ_s Σ_a D_demo(s,a) φ(s,a,k)
            return jnp.einsum("sa,sak->k", D_sa, feature_matrix)
        else:
            _ns = feature_matrix.shape[0]
            # For state-only features, compute state occupancy directly
            D_s, _ = self._compute_empirical_state_occupancy(
                panel,
                n_states or _ns,
                n_actions or getattr(reward_fn, "num_actions", 1),
                discount=discount,
            )
            return D_s @ feature_matrix

    def _compute_expected_features(
        self,
        panel: Panel,
        policy: jnp.ndarray,
        reward_fn: BaseUtilityFunction,
        transitions: jnp.ndarray | None = None,
        initial_dist: jnp.ndarray | None = None,
        discount: float = 0.9,
    ) -> jnp.ndarray:
        """Compute expected feature expectations under the learned policy.

        Uses the occupancy-measure approach from Ziebart (2010) Algorithm 1
        and the imitation library (Gleave & Toyer 2022):

            μ_π = Σ_s D_π(s) Σ_a π(a|s) φ(s, a)

        where D_π(s) is the state visitation frequency under the current
        policy, computed via the forward pass. This correctly handles both
        action-dependent and state-only features — when φ(s,a) = φ(s),
        the policy weights still matter through D_π.

        Falls back to empirical-state iteration only if transitions are
        not available.

        Parameters
        ----------
        panel : Panel
            Panel data (used only as fallback if transitions unavailable).
        policy : jnp.ndarray
            Policy probabilities π(a|s), shape (n_states, n_actions).
        reward_fn : BaseUtilityFunction
            Reward function with feature matrix.
        transitions : jnp.ndarray, optional
            Transition matrices, shape (n_actions, n_states, n_states).
        initial_dist : jnp.ndarray, optional
            Initial state distribution.
        discount : float
            Discount factor for state visitation computation.

        Returns
        -------
        jnp.ndarray
            Expected features, shape (n_features,).
        """
        # Get feature matrix - handle both 2D and 3D cases
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
        else:
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        # --- Occupancy measure approach (correct for all feature types) ---
        # E_π[φ] = Σ_s D_π(s) Σ_a π(a|s) φ(s,a)
        if transitions is not None and initial_dist is not None:
            n_states = policy.shape[0]
            problem = DDCProblem(
                num_states=n_states,
                num_actions=policy.shape[1],
                discount_factor=discount,
            )
            d = self._compute_state_visitation(policy, transitions, problem, initial_dist)

            # Promote feature_matrix to match computation dtype (float64 for high gamma)
            fm = feature_matrix.astype(d.dtype)
            pol = policy.astype(d.dtype)
            if fm.ndim == 3:
                # E_π[φ] = Σ_s D_π(s) Σ_a π(a|s) φ(s,a,k)
                return jnp.einsum("s,sa,sak->k", d, pol, fm)
            else:
                # E_π[φ] = Σ_s D_π(s) φ(s,k)
                return d @ fm

        # --- Fallback: empirical state iteration (only works for action-dependent) ---
        total_obs = sum(len(traj) for traj in panel.trajectories)

        if feature_matrix.ndim == 3:
            all_states_np = np.asarray(panel.get_all_states()) if hasattr(panel.get_all_states(), '__len__') else np.concatenate([np.asarray(traj.states) for traj in panel.trajectories])
            all_states_idx = jnp.array(all_states_np)
            feature_sum = (policy[all_states_idx][:, :, None] * feature_matrix[all_states_idx]).sum(axis=(0, 1))
            if total_obs > 0:
                return feature_sum / total_obs
            return feature_sum
        else:
            all_states = panel.get_all_states()
            all_states_idx = jnp.array(np.asarray(all_states))
            feature_sum = feature_matrix[all_states_idx, :].sum(axis=0)
            if total_obs > 0:
                return feature_sum / total_obs
            return feature_sum

    def _compute_expected_features_finite_horizon(
        self,
        panel: Panel,
        policy: jnp.ndarray,
        reward_fn: BaseUtilityFunction,
        num_periods: int,
        transitions: jnp.ndarray | None = None,
        initial_dist: jnp.ndarray | None = None,
        discount: float = 0.95,
    ) -> jnp.ndarray:
        """Compute expected features using finite-horizon occupancy measures.

        Uses forward-pass state visitation with time-indexed policies:
            D_0(s) = ρ_0(s)
            D_{t+1}(s') = Σ_s Σ_a D_t(s) π_t(a|s) P(s'|s,a)
            E_π[φ] = Σ_t γ^t Σ_s D_t(s) Σ_a π_t(a|s) φ(s,a,k)

        Parameters
        ----------
        policy : jnp.ndarray
            Time-indexed policy, shape (num_periods, n_states, n_actions).
        transitions : jnp.ndarray, optional
            Transition matrices, shape (n_actions, n_states, n_states).
        initial_dist : jnp.ndarray, optional
            Initial state distribution ρ_0, shape (n_states,).
        discount : float
            Discount factor γ.
        """
        if hasattr(reward_fn, 'feature_matrix'):
            feature_matrix = reward_fn.feature_matrix
        elif hasattr(reward_fn, 'state_features'):
            feature_matrix = reward_fn.state_features
        else:
            raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        n_states = policy.shape[1]
        n_actions = policy.shape[2]

        if initial_dist is None:
            D_t = jnp.ones(n_states) / n_states
        else:
            D_t = jnp.array(initial_dist)

        # Forward pass: accumulate discounted occupancy measures
        feature_sum = jnp.zeros(feature_matrix.shape[-1])

        for t in range(num_periods):
            if feature_matrix.ndim == 3:
                # E_π[φ]_t = γ^t Σ_s D_t(s) Σ_a π_t(a|s) φ(s,a,k)
                feature_sum = feature_sum + (discount ** t) * jnp.einsum(
                    "s,sa,sak->k", D_t, policy[t], feature_matrix
                )
            else:
                # State-only: E_π[φ]_t = γ^t Σ_s D_t(s) φ(s,k)
                feature_sum = feature_sum + (discount ** t) * (D_t @ feature_matrix)

            # Advance state distribution: D_{t+1}(s') = Σ_s Σ_a D_t(s) π_t(a|s) P(s'|s,a)
            if transitions is not None and t < num_periods - 1:
                P_pi_t = jnp.einsum("sa,ast->st", policy[t], transitions)
                D_t = P_pi_t.T @ D_t

        # Normalize by total discounted weight
        total_weight = sum(discount ** t for t in range(num_periods))
        return feature_sum / total_weight

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute initial state distribution from data."""
        init_states_list = [
            int(traj.states[0]) for traj in panel.trajectories if len(traj) > 0
        ]
        counts_np = np.zeros(n_states, dtype=np.float32)
        init_states_np = np.array(init_states_list, dtype=np.int64)
        np.add.at(counts_np, init_states_np, 1.0)

        counts = jnp.array(counts_np)
        if float(counts.sum()) > 0:
            return counts / counts.sum()
        return jnp.ones(n_states) / n_states

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        true_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run MCE IRL optimization.

        Parameters
        ----------
        true_params : jnp.ndarray, optional
            True parameters for debugging. If provided, RMSE is shown in progress bar.
        """
        start_time = time.time()

        reward_fn = utility
        # Use float64 for the Bellman operator to handle high discount factors
        # (condition number of (I - beta*P) ~ 1/(1-beta), so float32 is insufficient
        # for beta > 0.99)
        transitions_f64 = transitions.astype(jnp.float64)
        operator = SoftBellmanOperator(problem, transitions_f64)

        # Initialize parameters
        if initial_params is None:
            params = reward_fn.get_initial_parameters()
        else:
            params = jnp.array(initial_params)

        # Compute empirical features (constant) — all in float64 for precision
        empirical_features = self._compute_empirical_features(
            panel,
            reward_fn,
            problem.num_states,
            problem.num_actions,
            discount=problem.discount_factor,
        ).astype(jnp.float64)
        initial_dist = self._compute_initial_distribution(panel, problem.num_states).astype(jnp.float64)

        # Compute empirical state occupancy (constant) for occupancy distance check
        # Following Gleave & Toyer (2022): convergence when max|D_demo - D_policy| < tol
        D_demo, D_sa_demo = self._compute_empirical_state_occupancy(
            panel,
            problem.num_states,
            problem.num_actions,
            discount=problem.discount_factor,
        )
        D_demo = D_demo.astype(jnp.float64)
        D_sa_demo = D_sa_demo.astype(jnp.float64)

        self._log(f"Empirical features: {empirical_features}")
        self._log(f"Initial distribution entropy: {-(initial_dist * jnp.log(initial_dist + 1e-10)).sum():.3f}")

        # Tracking
        n_function_evals = 0
        inner_not_converged = 0

        # ── L-BFGS-B path (scipy) ──
        if self.config.optimizer in ("L-BFGS-B", "BFGS"):
            self._log(f"Starting MCE IRL with {self.config.optimizer}")

            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()

            prev_V = [None]  # mutable container for warm-starting
            gamma = problem.discount_factor
            sigma = problem.scale_parameter

            def neg_ll_and_grad(theta_np):
                nonlocal n_function_evals, inner_not_converged
                n_function_evals += 1
                theta = jnp.array(theta_np, dtype=jnp.float64)
                # Compute reward in float64 for numerical precision with high discount factors
                reward_matrix = reward_fn.compute(theta.astype(jnp.float32)).astype(jnp.float64)
                V, policy, converged = self._soft_value_iteration(
                    operator, reward_matrix, V_init=prev_V[0],
                )
                prev_V[0] = jnp.array(V)
                if not converged:
                    inner_not_converged += 1
                # Log-likelihood: LL = Σ_t log π(a_t|s_t)
                log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
                ll = float(log_probs[all_states, all_actions].sum())
                # Analytical gradient via implicit differentiation (same as NFXP):
                # dV/dθ_k = (I - γ P_π)^{-1} [Σ_a π(a|s) φ(s,a,k)]
                # dLL/dθ_k = (1/σ) Σ_t [dQ_k(s_t,a_t) - dV_k(s_t)]
                fm = reward_fn.feature_matrix.astype(jnp.float64) if hasattr(reward_fn, 'feature_matrix') else reward_fn.state_features.astype(jnp.float64)
                n_params = fm.shape[-1]
                n_states = problem.num_states
                # Build P_π: policy-weighted transition matrix
                P_pi = jnp.einsum("sa,asn->sn", policy, transitions_f64)
                # Solve for dV/dθ (one linear system per parameter)
                A = jnp.eye(n_states, dtype=jnp.float64) - gamma * P_pi
                grad_ll = np.zeros(n_params, dtype=np.float64)
                for k in range(n_params):
                    if fm.ndim == 3:
                        pi_phi_k = (policy * fm[:, :, k]).sum(axis=1)  # (S,)
                    else:
                        pi_phi_k = fm[:, k]  # (S,)
                    dV_k = jnp.linalg.solve(A, pi_phi_k)
                    # dQ_k(s,a) = φ(s,a,k) + γ Σ_s' P(s'|s,a) dV_k(s')
                    EV_k = jnp.einsum("asn,n->sa", transitions_f64, dV_k)  # (S, A)
                    if fm.ndim == 3:
                        dQ_k = fm[:, :, k] + gamma * EV_k  # (S, A)
                    else:
                        dQ_k = fm[:, k][:, None] + gamma * EV_k  # (S, A)
                    # ∂LL/∂θ_k = (1/σ) Σ_t [dQ_k(s_t,a_t) - Σ_a π(a|s_t) dQ_k(s_t,a)]
                    dQ_obs = dQ_k[all_states, all_actions]  # (N,)
                    dV_obs = (policy[all_states] * dQ_k[all_states]).sum(axis=1)  # (N,)
                    grad_ll[k] = float(((dQ_obs - dV_obs) / sigma).sum())
                if n_function_evals % 10 == 0:
                    self._log(f"Eval {n_function_evals}: NLL={-ll:.2f}, ||grad||={np.linalg.norm(grad_ll):.6f}")
                return -ll, -grad_ll

            result_opt = minimize_lbfgsb(
                neg_ll_and_grad,
                jnp.asarray(params, dtype=jnp.float64),
                bounds=None,
                maxiter=self.config.outer_max_iter,
                tol=self.config.outer_tol,
                verbose=self.config.verbose,
                desc="MCE-IRL L-BFGS-B",
                value_and_grad=True,
                jit=False,
            )
            final_params = jnp.array(result_opt.x, dtype=jnp.float32)
            converged = result_opt.success

            # Final solution
            reward_matrix = reward_fn.compute(final_params).astype(jnp.float64)
            V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
            D = self._compute_state_visitation(policy, transitions_f64, problem, initial_dist)
            D_sa = D[:, None] * policy
            occupancy_moment_residual = float(jnp.max(jnp.abs(D_sa_demo - D_sa)))
            final_expected = self._compute_expected_features(
                panel, policy, reward_fn,
                transitions=transitions_f64, initial_dist=initial_dist,
                discount=problem.discount_factor,
            )
            feature_diff = float(jnp.linalg.norm(empirical_features - final_expected))
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
            ll = float(log_probs[all_states, all_actions].sum())

            self._log(f"L-BFGS-B complete: NLL={-ll:.2f}, converged={converged}")
            if inner_not_converged > 0:
                self._log(f"Warning: Inner loop did not converge {inner_not_converged} times")

            # Compute Hessian for standard errors if requested
            hessian = None
            if self.config.compute_se and self.config.se_method != "bootstrap":
                self._log(f"Computing numerical Hessian for standard errors")
                hessian = self._numerical_hessian(
                    final_params, panel, reward_fn, problem, transitions_f64, initial_dist
                )

            optimization_time = time.time() - start_time
            return EstimationResult(
                parameters=final_params,
                log_likelihood=ll,
                value_function=V,
                policy=policy,
                hessian=hessian,
                gradient_contributions=None,
                converged=converged,
                num_iterations=n_function_evals,
                num_function_evals=n_function_evals,
                num_inner_iterations=0,
                message=result_opt.message,
                optimization_time=optimization_time,
                metadata={
                    "optimizer": self.config.optimizer,
                    "empirical_features": np.asarray(empirical_features).tolist(),
                    "final_expected_features": np.asarray(final_expected).tolist(),
                    "feature_diff": feature_diff,
                    "feature_difference": feature_diff,
                    "occupancy_moment_residual": occupancy_moment_residual,
                    "state_visitation": np.asarray(D).tolist(),
                    "state_action_visitation": np.asarray(D_sa).tolist(),
                },
            )

        # ── Root-finding path ──
        # Solve the feature-matching equation mu_D - mu_pi(theta) = 0 directly
        # via scipy.optimize.root. This is the explicit feature-matching solver
        # described in the JSS paper Versions paragraph and corresponds to the
        # primal stationarity condition of MCE IRL (Ziebart 2010, eq 3.1).
        if self.config.optimizer == "root":
            from scipy.optimize import root as _scipy_root

            self._log("Starting MCE IRL with root-finding (feature matching)")
            finite_horizon = problem.num_periods is not None
            num_periods = problem.num_periods

            def feature_residual(theta_np: np.ndarray) -> np.ndarray:
                theta = jnp.asarray(theta_np, dtype=jnp.float64)
                reward_matrix = reward_fn.compute(theta).astype(jnp.float64)
                V, policy, _ = self._soft_value_iteration(
                    operator, reward_matrix, num_periods=num_periods,
                )
                if finite_horizon:
                    expected = self._compute_expected_features_finite_horizon(
                        panel, policy, reward_fn, num_periods,
                        transitions=transitions_f64, initial_dist=initial_dist,
                        discount=problem.discount_factor,
                    )
                else:
                    expected = self._compute_expected_features(
                        panel, policy, reward_fn,
                        transitions=transitions_f64,
                        initial_dist=initial_dist,
                        discount=problem.discount_factor,
                    )
                return np.asarray(empirical_features - expected, dtype=np.float64)

            sol = _scipy_root(
                feature_residual,
                np.asarray(params, dtype=np.float64),
                method="hybr",
                tol=self.config.outer_tol,
                options={"maxfev": self.config.outer_max_iter * (len(params) + 1)},
            )
            params = jnp.asarray(sol.x, dtype=jnp.float64)
            converged = bool(sol.success)
            n_function_evals = int(getattr(sol, "nfev", 0))

            reward_matrix = reward_fn.compute(params).astype(jnp.float64)
            V, policy, _ = self._soft_value_iteration(
                operator, reward_matrix, num_periods=num_periods,
            )
            if finite_horizon:
                final_expected = self._compute_expected_features_finite_horizon(
                    panel, policy, reward_fn, num_periods,
                    transitions=transitions_f64, initial_dist=initial_dist,
                    discount=problem.discount_factor,
                )
                D = jnp.zeros(problem.num_states, dtype=jnp.float64)
                D_sa = jnp.zeros((problem.num_states, problem.num_actions), dtype=jnp.float64)
                occupancy_moment_residual = None
                ll = 0.0
                V = V[0] if V.ndim > 1 else V
                policy = policy[0] if policy.ndim > 2 else policy
            else:
                final_expected = self._compute_expected_features(
                    panel, policy, reward_fn,
                    transitions=transitions_f64, initial_dist=initial_dist,
                    discount=problem.discount_factor,
                )
                D = self._compute_state_visitation(
                    policy,
                    transitions_f64,
                    problem,
                    initial_dist,
                )
                D_sa = D[:, None] * policy
                occupancy_moment_residual = float(jnp.max(jnp.abs(D_sa_demo - D_sa)))
                log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
                ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())
            feature_diff = float(jnp.linalg.norm(empirical_features - final_expected))
            elapsed = time.time() - start_time
            self._log(
                f"Root-finding done in {elapsed:.2f}s, "
                f"||mu_D - mu_pi|| = {feature_diff:.6f}"
            )
            return EstimationResult(
                parameters=params,
                log_likelihood=ll,
                value_function=V,
                policy=policy,
                converged=converged,
                num_iterations=n_function_evals,
                metadata={
                    "optimizer": self.config.optimizer,
                    "estimator": "MCE-IRL (root)",
                    "elapsed_seconds": elapsed,
                    "empirical_features": np.asarray(empirical_features).tolist(),
                    "final_expected_features": np.asarray(final_expected).tolist(),
                    "feature_diff": feature_diff,
                    "feature_difference": feature_diff,
                    "occupancy_moment_residual": occupancy_moment_residual,
                    "state_visitation": np.asarray(D).tolist(),
                    "state_action_visitation": np.asarray(D_sa).tolist(),
                },
            )

        # ── Adam/SGD path (gradient ascent) ──
        # Adam optimizer state
        if self.config.use_adam:
            m = jnp.zeros_like(params)  # First moment
            v = jnp.zeros_like(params)  # Second moment
            optimizer_name = "Adam"
        else:
            m = v = None
            optimizer_name = "SGD"

        # Run optimization using gradient ascent on log-likelihood
        # The gradient is: ∇L = μ_D - μ_π (for maximization)
        self._log(f"Starting MCE IRL with {optimizer_name} (lr={self.config.learning_rate})")

        best_obj = float('inf')
        best_params = jnp.array(params)
        patience_counter = 0
        max_patience = 20

        # Use tqdm for progress tracking
        pbar = tqdm(
            range(self.config.outer_max_iter),
            desc="MCE IRL",
            disable=not self.config.verbose,
            leave=True,
        )

        finite_horizon = problem.num_periods is not None
        num_periods = problem.num_periods

        prev_V_grad = None  # warm-start for gradient path

        for i in pbar:
            # Forward and backward passes
            reward_matrix = reward_fn.compute(params).astype(jnp.float64)
            V, policy, inner_converged = self._soft_value_iteration(
                operator, reward_matrix, num_periods=num_periods,
                V_init=prev_V_grad,
            )
            prev_V_grad = jnp.array(V) if not finite_horizon else None
            if not inner_converged:
                inner_not_converged += 1

            if finite_horizon:
                # For finite horizon, compute expected features using time-indexed policy
                # Average policy across periods weighted by empirical period distribution
                expected_features = self._compute_expected_features_finite_horizon(
                    panel, policy, reward_fn, num_periods,
                    transitions=transitions_f64, initial_dist=initial_dist,
                    discount=problem.discount_factor,
                )
            else:
                expected_features = self._compute_expected_features(
                    panel, policy, reward_fn,
                    transitions=transitions_f64,
                    initial_dist=initial_dist,
                    discount=problem.discount_factor,
                )

            # Feature matching gradient: μ_D - μ_π
            # We want to INCREASE params in direction where μ_D > μ_π
            gradient = empirical_features - expected_features  # Gradient ascent direction

            # Gradient clipping (prevents divergence when rewards approach zero)
            grad_norm = float(jnp.linalg.norm(gradient))
            if self.config.gradient_clip > 0 and grad_norm > self.config.gradient_clip:
                gradient = gradient * (self.config.gradient_clip / grad_norm)
                grad_norm = self.config.gradient_clip

            # Objective: ||μ_D - μ_π||^2
            obj = float(0.5 * jnp.sum((empirical_features - expected_features) ** 2))

            # Occupancy distance: max|D_demo - D_policy| (Gleave & Toyer 2022)
            # Only computed for infinite horizon where state visitation is meaningful
            if not finite_horizon:
                D_policy = self._compute_state_visitation(
                    policy, transitions_f64, problem, initial_dist
                )
                occ_dist = float(jnp.max(jnp.abs(D_demo - D_policy)))
            else:
                occ_dist = float('inf')

            # Track best
            if obj < best_obj:
                best_obj = obj
                best_params = jnp.array(params)
                patience_counter = 0
            else:
                patience_counter += 1

            # Update progress bar with metrics
            postfix = {
                "obj": f"{obj:.6f}",
                "||grad||": f"{grad_norm:.4f}",
                "occ": f"{occ_dist:.4f}",
            }
            if true_params is not None:
                rmse = float(jnp.sqrt(jnp.mean((params - true_params) ** 2)))
                postfix["RMSE"] = f"{rmse:.6f}"
            pbar.set_postfix(postfix)

            # Check convergence: gradient norm OR occupancy distance
            if grad_norm < self.config.outer_tol:
                converged = True
                break

            if occ_dist < self.config.occupancy_tol:
                converged = True
                self._log(f"Converged by occupancy distance: {occ_dist:.6f}")
                break

            if patience_counter > max_patience:
                pbar.set_description("MCE IRL (early stop)")
                break

            # Gradient step with Adam or SGD
            if self.config.use_adam:
                # Adam update (Kingma & Ba, 2014)
                t = i + 1  # Timestep (1-indexed)
                m = self.config.adam_beta1 * m + (1 - self.config.adam_beta1) * gradient
                v = self.config.adam_beta2 * v + (1 - self.config.adam_beta2) * (gradient ** 2)
                # Bias correction
                m_hat = m / (1 - self.config.adam_beta1 ** t)
                v_hat = v / (1 - self.config.adam_beta2 ** t)
                # Update
                params = params + self.config.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.config.adam_eps)
            else:
                # Simple SGD
                params = params + self.config.learning_rate * gradient

            n_function_evals += 1

        pbar.close()
        final_params = best_params
        converged = grad_norm < self.config.outer_tol if grad_norm else False

        # Final solution
        reward_matrix = reward_fn.compute(final_params).astype(jnp.float64)
        V, policy, _ = self._soft_value_iteration(
            operator, reward_matrix, num_periods=num_periods
        )

        if finite_horizon:
            final_expected = self._compute_expected_features_finite_horizon(
                panel, policy, reward_fn, num_periods,
                transitions=transitions_f64, initial_dist=initial_dist,
                discount=problem.discount_factor,
            )
            D = jnp.zeros(problem.num_states)
            D_sa = jnp.zeros((problem.num_states, problem.num_actions))
            occupancy_moment_residual = None

            # Finite-horizon LL: use period-specific policies
            sigma = problem.scale_parameter
            fh_result = backward_induction(operator, reward_matrix, num_periods)
            log_policy = jax.nn.log_softmax(fh_result.Q / sigma, axis=-1)  # (T, S, A)
            ll = 0.0
            for traj in panel.trajectories:
                for t in range(len(traj.states)):
                    period = min(t, num_periods - 1)
                    s = int(traj.states[t])
                    a = int(traj.actions[t])
                    ll += float(log_policy[period, s, a])

            # Flatten to period-0 for EstimationResult (base class expects 2D)
            V = V[0] if V.ndim > 1 else V
            policy = policy[0] if policy.ndim > 2 else policy
        else:
            D = self._compute_state_visitation(policy, transitions_f64, problem, initial_dist)
            D_sa = D[:, None] * policy
            occupancy_moment_residual = float(jnp.max(jnp.abs(D_sa_demo - D_sa)))
            final_expected = self._compute_expected_features(
                panel, policy, reward_fn,
                transitions=transitions_f64, initial_dist=initial_dist,
                discount=problem.discount_factor,
            )
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
            ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())

        feature_diff = float(jnp.linalg.norm(empirical_features - final_expected))

        # Inference
        hessian = None
        gradient_contributions = None
        standard_errors = None

        if self.config.compute_se:
            self._log(f"Computing standard errors via {self.config.se_method}")

            if self.config.se_method == "bootstrap":
                standard_errors = self._bootstrap_inference(
                    panel, reward_fn, problem, transitions_f64, final_params, initial_dist
                )
            else:
                # Numerical Hessian
                hessian = self._numerical_hessian(
                    final_params, panel, reward_fn, problem, transitions_f64, initial_dist
                )

        optimization_time = time.time() - start_time

        self._log(f"Optimization complete: feature_diff={feature_diff:.6f}, LL={ll:.2f}")
        if inner_not_converged > 0:
            self._log(f"Warning: Inner loop did not converge {inner_not_converged} times")

        return EstimationResult(
            parameters=final_params,
            log_likelihood=ll,
            value_function=V,
            policy=policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=n_function_evals,
            num_function_evals=n_function_evals,
            num_inner_iterations=0,
            message="",
            optimization_time=optimization_time,
            metadata={
                "optimizer": self.config.optimizer,
                "empirical_features": np.asarray(empirical_features).tolist(),
                "final_expected_features": np.asarray(final_expected).tolist(),
                "feature_difference": feature_diff,
                "feature_diff": feature_diff,
                "occupancy_moment_residual": occupancy_moment_residual,
                "state_visitation": np.asarray(D).tolist(),
                "state_action_visitation": np.asarray(D_sa).tolist(),
                "inner_not_converged": inner_not_converged,
                "standard_errors": np.asarray(standard_errors).tolist() if standard_errors is not None else None,
            },
        )

    def _bootstrap_inference(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        point_estimate: jnp.ndarray,
        initial_dist: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute standard errors via bootstrap.

        Resamples trajectories and re-estimates parameters to get
        sampling distribution.
        """
        n_params = len(point_estimate)
        bootstrap_estimates = np.zeros((self.config.n_bootstrap, n_params))

        trajectories = panel.trajectories
        n_traj = len(trajectories)

        operator = SoftBellmanOperator(problem, transitions)

        for b in range(self.config.n_bootstrap):
            # Resample trajectories with replacement
            indices = np.random.choice(n_traj, size=n_traj, replace=True)
            boot_trajectories = [trajectories[i] for i in indices]
            boot_panel = Panel(trajectories=boot_trajectories)

            # Compute empirical features for bootstrap sample
            empirical_features = self._compute_empirical_features(
                boot_panel,
                reward_fn,
                problem.num_states,
                problem.num_actions,
                discount=problem.discount_factor,
            )
            boot_initial = self._compute_initial_distribution(boot_panel, problem.num_states)

            # Quick optimization from point estimate
            params = jnp.array(point_estimate)

            for _ in range(50):  # Fewer iterations for bootstrap
                reward_matrix = reward_fn.compute(params).astype(jnp.float64)
                V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
                expected_features = self._compute_expected_features(
                    boot_panel, policy, reward_fn,
                    transitions=transitions, initial_dist=boot_initial,
                    discount=problem.discount_factor,
                )

                gradient = empirical_features - expected_features  # μ_D - μ_π (ascent)
                params = params + 0.1 * gradient

                if float(jnp.linalg.norm(gradient)) < 0.01:
                    break

            bootstrap_estimates[b] = np.asarray(params)

            if self.config.verbose and (b + 1) % 20 == 0:
                self._log(f"Bootstrap {b + 1}/{self.config.n_bootstrap}")

        # Standard errors = std of bootstrap estimates
        standard_errors = jnp.array(bootstrap_estimates.std(axis=0))

        return standard_errors

    def _numerical_hessian(
        self,
        params: jnp.ndarray,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_dist: jnp.ndarray,
        eps: float = 1e-3,
    ) -> jnp.ndarray:
        """Compute numerical Hessian of the log-likelihood.

        Uses central differences with adaptive step size for numerical stability.
        The Hessian at a maximum should be negative semi-definite, so we ensure
        the returned Hessian has this property by projecting onto the negative
        semi-definite cone if needed.

        Parameters
        ----------
        params : jnp.ndarray
            Parameter vector at which to compute Hessian.
        panel : Panel
            Panel data for computing log-likelihood.
        reward_fn : LinearReward
            Reward function specification.
        problem : DDCProblem
            Problem specification.
        transitions : jnp.ndarray
            Transition matrices.
        initial_dist : jnp.ndarray
            Initial state distribution (unused but kept for API consistency).
        eps : float
            Step size for finite differences. Default 1e-3 for stability.

        Returns
        -------
        jnp.ndarray
            Hessian matrix, shape (n_params, n_params).
            Guaranteed to be negative semi-definite for valid inference.

        Notes
        -----
        Step size selection follows Gill, Murray, and Wright (1981) guidance:
        - Use adaptive step h_i = max(eps, min(|params[i]| * eps, 0.1))
        - Lower bound (eps) prevents division by zero for zero-valued parameters
        - Upper bound (0.1) prevents excessively large steps for large parameters
          that would introduce truncation error and numerical instability
        - The default eps=1e-3 balances truncation error (favors larger h)
          against rounding error (favors smaller h) for float32 precision
        """
        operator = SoftBellmanOperator(problem, transitions)
        n_params = len(params)
        hessian_np = np.zeros((n_params, n_params), dtype=np.float64)
        params_np = np.asarray(params, dtype=np.float64)

        def ll_at(p_np):
            p = jnp.array(p_np, dtype=jnp.float32)
            reward_matrix = reward_fn.compute(p).astype(jnp.float64)
            V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)

            ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())
            return ll

        # Compute Hessian using central differences
        # Use adaptive step size based on parameter magnitude with bounds
        for i in range(n_params):
            # Adaptive step: larger for larger params, bounded between eps and 0.1
            h_i = max(eps, min(abs(params_np[i]) * eps, 0.1))

            for j in range(i, n_params):
                h_j = max(eps, min(abs(params_np[j]) * eps, 0.1))

                if i == j:
                    # Diagonal: use standard 2nd derivative formula
                    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
                    p_plus = params_np.copy()
                    p_plus[i] += h_i
                    p_minus = params_np.copy()
                    p_minus[i] -= h_i

                    ll_plus = ll_at(p_plus)
                    ll_0 = ll_at(params_np)
                    ll_minus = ll_at(p_minus)

                    h_ii = (ll_plus - 2 * ll_0 + ll_minus) / (h_i * h_i)
                    hessian_np[i, i] = h_ii
                else:
                    # Off-diagonal: use 4-point formula for mixed partial
                    p_pp = params_np.copy()
                    p_pp[i] += h_i
                    p_pp[j] += h_j

                    p_pm = params_np.copy()
                    p_pm[i] += h_i
                    p_pm[j] -= h_j

                    p_mp = params_np.copy()
                    p_mp[i] -= h_i
                    p_mp[j] += h_j

                    p_mm = params_np.copy()
                    p_mm[i] -= h_i
                    p_mm[j] -= h_j

                    h_ij = (ll_at(p_pp) - ll_at(p_pm) - ll_at(p_mp) + ll_at(p_mm)) / (4 * h_i * h_j)
                    hessian_np[i, j] = h_ij
                    hessian_np[j, i] = h_ij

        hessian = jnp.array(hessian_np)

        # Ensure Hessian is negative semi-definite (required at a maximum)
        # If not, project onto negative semi-definite cone
        eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
        if bool(jnp.any(eigenvalues > 0)):
            # Some eigenvalues are positive - not at a proper maximum
            # Project to negative semi-definite by clamping positive eigenvalues
            self._log("Warning: Hessian not negative semi-definite, projecting")
            eigenvalues_clamped = jnp.minimum(eigenvalues, -1e-8)
            hessian = eigenvectors @ jnp.diag(eigenvalues_clamped) @ eigenvectors.T

        return hessian
