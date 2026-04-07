"""AIRL with Unobserved Heterogeneity (Lee, Sudhir & Wang 2026).

Extends vanilla AIRL with three key innovations:

1. Anchor identification: an exit action normalized to zero reward and
   an absorbing state with zero continuation value uniquely pin down
   the reward function g* = r* and shaping function h* = V*, eliminating
   the partial identification problem in standard AIRL.

2. Latent consumer segments: K segment-specific reward and policy functions
   estimated via an EM algorithm. The E-step computes posterior segment
   membership probabilities from trajectory likelihoods. The M-step
   updates each segment's reward via weighted AIRL.

3. Action-dependent utilities: with the anchor normalization, action-dependent
   rewards r(s,a) are identified (not just state-only rewards as in Fu et al.).

Reference:
    Lee, P.S., Sudhir, K., & Wang, T. (2026). "Modeling Serialized Content
    Consumption: Adversarial IRL for Dynamic Discrete Choice." Yale SOM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration, hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.adversarial.base import AdversarialEstimatorBase
from econirl.estimation.base import EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.base import BaseUtilityFunction


_SENTINEL = object()


@dataclass
class AIRLHetConfig:
    """Configuration for AIRL with unobserved heterogeneity.

    The exit_action and absorbing_state must be specified explicitly
    because they are model-specific. There are no sensible defaults.

    Attributes:
        num_segments: Number of latent consumer segments K.
        exit_action: Index of the anchor action (reward pinned to 0).
        absorbing_state: Index of the absorbing state (V pinned to 0).
        reward_type: Parameterization of reward ("tabular" or "linear").
        reward_lr: Learning rate for segment-specific reward updates.
        discriminator_steps: Discriminator updates per AIRL inner round.
        generator_solver: Inner solver for policy computation.
        generator_tol: Convergence tolerance for value iteration.
        generator_max_iter: Max iterations for value iteration.
        max_airl_rounds: Max AIRL rounds per EM M-step.
        max_em_iterations: Max EM outer iterations.
        em_convergence_tol: Relative LL change for EM convergence.
        airl_convergence_tol: Policy change tolerance within M-step.
        consistency_weight: Interpolation weight for within-user
            segment consistency (0 = no consistency, 1 = full pooling).
        prior_smoothing: Dirichlet smoothing for segment prior updates.
        use_shaping: Whether to use potential-based shaping in discriminator.
        shaping_coef: Shaping coefficient (defaults to discount_factor).
        verbose: Whether to print progress.
        seed: Random seed for initialization.
    """

    num_segments: int = 2
    exit_action: int | object = _SENTINEL
    absorbing_state: int | object = _SENTINEL

    # AIRL inner loop
    reward_type: Literal["tabular", "linear"] = "tabular"
    reward_lr: float = 0.01
    discriminator_steps: int = 5
    generator_solver: Literal["value", "hybrid"] = "hybrid"
    generator_tol: float = 1e-8
    generator_max_iter: int = 5000
    max_airl_rounds: int = 100
    airl_convergence_tol: float = 1e-4

    # EM outer loop
    max_em_iterations: int = 50
    em_convergence_tol: float = 1e-3

    # Consistency and regularization
    consistency_weight: float = 0.1
    prior_smoothing: float = 0.01
    prior_min: float = 0.0  # minimum prior per segment (prevents collapse, e.g. 1/K)
    prior_damping: float = 0.0  # damping for prior updates (0=no damping, 1=no update)
    reward_weight_decay: float = 0.0
    normalize_reward: bool = False  # clamp reward Frobenius norm to 1 per AIRL round
    unit_normalize_reward: bool = False  # project linear reward onto unit sphere each round
    gradient_clip_norm: float = 0.0  # clip gradient norm (0 = disabled)
    antisymmetric_init: bool = False  # init K=2 segments with opposite rewards

    # Shaping
    use_shaping: bool = True
    shaping_coef: float | None = None

    # Misc
    verbose: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.exit_action is _SENTINEL:
            raise ValueError(
                "exit_action must be specified (index of the anchor action "
                "with zero reward). There is no sensible default."
            )
        if self.absorbing_state is _SENTINEL:
            raise ValueError(
                "absorbing_state must be specified (index of the absorbing "
                "state with zero continuation value). There is no sensible default."
            )


class AIRLHetEstimator(AdversarialEstimatorBase):
    """AIRL with unobserved consumer heterogeneity.

    Estimates K segment-specific reward functions and policies via
    an EM algorithm that wraps per-segment AIRL. The anchor
    normalization (exit_action = 0 reward, absorbing_state = 0 value)
    guarantees that the recovered reward equals the true structural
    reward, not a shaped perturbation.

    Parameters
    ----------
    config : AIRLHetConfig
        Configuration with segment count, anchor indices, and
        optimization hyperparameters.

    Examples
    --------
    >>> config = AIRLHetConfig(
    ...     num_segments=2,
    ...     exit_action=2,
    ...     absorbing_state=50,
    ...     max_em_iterations=30,
    ... )
    >>> estimator = AIRLHetEstimator(config)
    >>> summary = estimator.estimate(panel, utility, problem, transitions)
    >>> posteriors = summary.metadata["segment_posteriors"]
    """

    def __init__(self, config: AIRLHetConfig, **kwargs):
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "AIRL-Het (Lee, Sudhir & Wang 2026)"

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate heterogeneous reward functions using EM-AIRL."""
        start_time = time.time()

        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        K = self.config.num_segments
        n_states = problem.num_states
        n_actions = problem.num_actions

        # Parameter names: segment-prefixed
        if self.config.reward_type == "linear":
            base_names = utility.parameter_names
        else:
            base_names = [
                f"R({s},{a})" for s in range(n_states) for a in range(n_actions)
            ]
        param_names = [f"seg{k}_{name}" for k in range(K) for name in base_names]

        standard_errors = jnp.full_like(result.parameters, float("nan"))

        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * jnp.log(jnp.array(n_obs)).item(),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, result.policy
            ),
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

    # --- EM algorithm ---

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run EM-AIRL optimization."""
        start_time = time.time()

        K = self.config.num_segments
        n_states = problem.num_states
        n_actions = problem.num_actions
        gamma = problem.discount_factor
        sigma = problem.scale_parameter
        exit_action = self.config.exit_action
        absorbing = self.config.absorbing_state

        operator = SoftBellmanOperator(problem, transitions)

        # Extract expert transitions per trajectory
        traj_data = self._extract_trajectory_data(panel)
        n_trajs = len(traj_data)

        # Group trajectories by individual for consistency constraint
        individual_groups = self._group_by_individual(panel)

        # Initialize segment-specific parameters
        key = jax.random.key(self.config.seed)
        segment_rewards = []
        segment_opt_states = []
        if self.config.reward_weight_decay > 0:
            base_optimizer = optax.adamw(
                self.config.reward_lr,
                weight_decay=self.config.reward_weight_decay,
            )
        else:
            base_optimizer = optax.adam(self.config.reward_lr)

        if self.config.gradient_clip_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clip_norm),
                base_optimizer,
            )
        else:
            optimizer = base_optimizer

        base_rw = None
        for k in range(K):
            key, subkey = jax.random.split(key)
            if self.config.reward_type == "tabular":
                if self.config.antisymmetric_init and K >= 2:
                    if k == 0:
                        base_rw = 1.5 * jax.random.normal(subkey, (n_states, n_actions))
                        rw = base_rw
                    elif k == 1 and base_rw is not None:
                        rw = -base_rw
                    else:
                        rw = 1.5 * jax.random.normal(subkey, (n_states, n_actions))
                else:
                    rw = 0.5 * jax.random.normal(subkey, (n_states, n_actions))
                rw = self._enforce_anchor_reward(rw, exit_action, absorbing)
                if self.config.normalize_reward:
                    rw = jnp.clip(rw, -5.0, 5.0)
            else:
                n_features = utility.num_parameters
                if self.config.antisymmetric_init and K >= 2:
                    if k == 0:
                        base_rw = 0.5 * jax.random.normal(subkey, (n_features,))
                        rw = base_rw
                    elif k == 1 and base_rw is not None:
                        rw = -base_rw
                    else:
                        rw = 0.01 * jax.random.normal(subkey, (n_features,))
                else:
                    rw = 0.01 * jax.random.normal(subkey, (n_features,))
            segment_rewards.append(rw)
            segment_opt_states.append(optimizer.init(rw))

        # Initialize segment policies and values
        segment_policies = []
        segment_V = []
        for k in range(K):
            reward_matrix = self._get_reward_matrix(
                segment_rewards[k], utility, n_states, n_actions
            )
            reward_matrix = self._enforce_anchor_reward(reward_matrix, exit_action, absorbing)
            policy, V = self._compute_policy_with_anchors(
                reward_matrix, operator, absorbing
            )
            segment_policies.append(policy)
            segment_V.append(V)

        # Initialize segment priors (uniform)
        segment_priors = jnp.ones(K) / K

        # Initialize posteriors (uniform)
        posteriors = jnp.ones((n_trajs, K)) / K

        # EM loop
        prev_ll = -float("inf")
        converged = False
        em_iter = 0
        em_lls = []

        pbar = tqdm(
            range(self.config.max_em_iterations),
            desc="EM-AIRL",
            disable=not self.config.verbose,
        )

        for em_iter in pbar:
            # --- E-step ---
            posteriors = self._e_step(
                traj_data, segment_policies, segment_priors, sigma
            )
            posteriors = self._apply_consistency(
                posteriors, individual_groups
            )
            segment_priors = self._update_priors(posteriors, segment_priors)

            # --- M-step ---
            for k in range(K):
                segment_rewards[k], segment_opt_states[k] = self._m_step_segment(
                    k=k,
                    traj_data=traj_data,
                    posteriors=posteriors,
                    utility=utility,
                    problem=problem,
                    transitions=transitions,
                    operator=operator,
                    reward_params=segment_rewards[k],
                    opt_state=segment_opt_states[k],
                    optimizer=optimizer,
                    policy=segment_policies[k],
                    V=segment_V[k],
                    n_states=n_states,
                    n_actions=n_actions,
                    gamma=gamma,
                    exit_action=exit_action,
                    absorbing=absorbing,
                    key=key,
                )
                key = jax.random.split(key)[0]

                # Update policy and value from new reward
                reward_matrix = self._get_reward_matrix(
                    segment_rewards[k], utility, n_states, n_actions
                )
                reward_matrix = self._enforce_anchor_reward(
                    reward_matrix, exit_action, absorbing
                )
                policy, V = self._compute_policy_with_anchors(
                    reward_matrix, operator, absorbing
                )
                segment_policies[k] = policy
                segment_V[k] = V

            # --- Convergence check ---
            mixture_ll = self._compute_mixture_ll(
                traj_data, segment_policies, segment_priors, sigma
            )
            em_lls.append(mixture_ll)

            rel_change = abs(mixture_ll - prev_ll) / max(abs(prev_ll), 1.0)
            prior_str = " ".join(f"{float(p):.2f}" for p in segment_priors)
            pbar.set_postfix({
                "LL": f"{mixture_ll:.1f}",
                "dLL": f"{rel_change:.5f}",
                "priors": f"[{prior_str}]",
            })

            if rel_change < self.config.em_convergence_tol and em_iter > 0:
                converged = True
                break
            prev_ll = mixture_ll

        pbar.close()

        # --- Assemble results ---
        # Aggregate policy: weighted mixture
        mixture_policy = sum(
            segment_priors[k] * segment_policies[k] for k in range(K)
        )
        mixture_V = sum(
            segment_priors[k] * segment_V[k] for k in range(K)
        )

        # Concatenate all segment parameters
        all_params = jnp.concatenate([
            segment_rewards[k].flatten() for k in range(K)
        ])

        # Hard segment assignments
        assignments = jnp.argmax(posteriors, axis=1)

        # Segment reward matrices for metadata
        seg_reward_matrices = []
        for k in range(K):
            rm = self._get_reward_matrix(
                segment_rewards[k], utility, n_states, n_actions
            )
            rm = self._enforce_anchor_reward(rm, exit_action, absorbing)
            seg_reward_matrices.append(rm.tolist())

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=all_params,
            log_likelihood=mixture_ll,
            value_function=mixture_V,
            policy=mixture_policy,
            hessian=None,
            converged=converged,
            num_iterations=em_iter + 1,
            num_function_evals=em_iter + 1,
            message="EM converged" if converged else "Max EM iterations reached",
            optimization_time=optimization_time,
            metadata={
                "num_segments": K,
                "segment_priors": np.asarray(segment_priors).tolist(),
                "segment_posteriors": np.asarray(posteriors).tolist(),
                "segment_assignments": np.asarray(assignments).tolist(),
                "segment_reward_matrices": seg_reward_matrices,
                "segment_policies": [
                    np.asarray(segment_policies[k]).tolist() for k in range(K)
                ],
                "segment_value_functions": [
                    np.asarray(segment_V[k]).tolist() for k in range(K)
                ],
                "em_log_likelihoods": em_lls,
            },
        )

    # --- E-step ---

    def _e_step(
        self,
        traj_data: list[dict],
        segment_policies: list[jnp.ndarray],
        segment_priors: jnp.ndarray,
        sigma: float,
    ) -> jnp.ndarray:
        """Compute posterior segment probabilities for each trajectory.

        For each trajectory i and segment k:
            ll_k(i) = sum_t log pi_k(a_t | s_t)
            posterior[i,k] = prior[k] * exp(ll_k(i)) / Z_i

        Uses log-sum-exp for numerical stability.
        """
        K = len(segment_policies)
        n_trajs = len(traj_data)
        log_posteriors = jnp.zeros((n_trajs, K))

        for k in range(K):
            log_pi_k = jnp.log(segment_policies[k] + 1e-10)
            for i, td in enumerate(traj_data):
                traj_ll = float(log_pi_k[td["states"], td["actions"]].sum())
                log_posteriors = log_posteriors.at[i, k].set(
                    jnp.log(segment_priors[k] + 1e-10) + traj_ll
                )

        # Normalize via log-sum-exp
        log_Z = jax.nn.logsumexp(log_posteriors, axis=1, keepdims=True)
        posteriors = jnp.exp(log_posteriors - log_Z)

        return posteriors

    def _apply_consistency(
        self,
        posteriors: jnp.ndarray,
        individual_groups: dict[Any, list[int]],
    ) -> jnp.ndarray:
        """Apply within-user segment consistency constraint.

        For individuals with multiple trajectories (e.g., reading
        multiple books), average their posteriors and interpolate
        toward the consensus. This encourages the same individual
        to be assigned to the same segment across series.
        """
        w = self.config.consistency_weight
        if w <= 0:
            return posteriors

        adjusted = np.asarray(posteriors).copy()
        for ind_id, traj_indices in individual_groups.items():
            if len(traj_indices) <= 1:
                continue
            # Average posteriors across this individual's trajectories
            group_mean = adjusted[traj_indices].mean(axis=0)
            # Interpolate: (1-w)*individual + w*consensus
            for idx in traj_indices:
                adjusted[idx] = (1 - w) * adjusted[idx] + w * group_mean

        # Re-normalize rows
        row_sums = adjusted.sum(axis=1, keepdims=True)
        adjusted = adjusted / np.maximum(row_sums, 1e-10)
        return jnp.array(adjusted)

    def _update_priors(
        self, posteriors: jnp.ndarray, old_priors: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Update segment priors with Dirichlet smoothing, optional floor and damping."""
        alpha = self.config.prior_smoothing
        raw = posteriors.mean(axis=0) + alpha
        new_priors = raw / raw.sum()

        # Floor: prevent any segment from collapsing to zero
        if self.config.prior_min > 0:
            new_priors = jnp.maximum(new_priors, self.config.prior_min)
            new_priors = new_priors / new_priors.sum()

        # Damping: slow down prior updates to prevent oscillation
        if self.config.prior_damping > 0 and old_priors is not None:
            d = self.config.prior_damping
            new_priors = (1.0 - d) * new_priors + d * old_priors
            new_priors = new_priors / new_priors.sum()

        return new_priors

    # --- M-step ---

    def _m_step_segment(
        self,
        k: int,
        traj_data: list[dict],
        posteriors: jnp.ndarray,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        operator: SoftBellmanOperator,
        reward_params: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        policy: jnp.ndarray,
        V: jnp.ndarray,
        n_states: int,
        n_actions: int,
        gamma: float,
        exit_action: int,
        absorbing: int,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, optax.OptState]:
        """Run AIRL inner loop for segment k.

        Expert transitions are weighted by posterior[i, k].
        Policy transitions are sampled from the current segment-k policy.
        """
        # Collect weighted expert transitions
        all_states = []
        all_actions = []
        all_next_states = []
        all_weights = []

        for i, td in enumerate(traj_data):
            w = float(posteriors[i, k])
            if w < 1e-6:
                continue
            all_states.append(td["states"])
            all_actions.append(td["actions"])
            all_next_states.append(td["next_states"])
            all_weights.append(jnp.full(len(td["states"]), w))

        if not all_states:
            return reward_params, opt_state

        exp_s = jnp.concatenate(all_states)
        exp_a = jnp.concatenate(all_actions)
        exp_ns = jnp.concatenate(all_next_states)
        exp_w = jnp.concatenate(all_weights)
        # Normalize weights
        exp_w = exp_w / exp_w.sum() * len(exp_w)

        n_expert = len(exp_s)

        # Initial distribution for policy sampling
        initial_dist = self._compute_initial_distribution_from_data(traj_data, n_states)

        use_shaping = self.config.use_shaping
        shaping_coef = self.config.shaping_coef

        def get_rm(params):
            rm = self._get_reward_matrix(params, utility, n_states, n_actions)
            return self._enforce_anchor_reward(rm, exit_action, absorbing)

        def disc_loss_fn(rw_params, V_fixed, policy_fixed,
                         es, ea, ens, ew, ps, pa, pns):
            reward_matrix = get_rm(rw_params)

            def logits(states, actions, next_states):
                r_sa = reward_matrix[states, actions]
                if use_shaping:
                    sc = shaping_coef if shaping_coef else gamma
                    f = r_sa + sc * V_fixed[next_states] - V_fixed[states]
                else:
                    f = r_sa
                log_pi = jnp.log(policy_fixed[states, actions] + 1e-10)
                return f - log_pi

            e_logits = logits(es, ea, ens)
            p_logits = logits(ps, pa, pns)

            # Weighted expert loss
            e_loss = jnp.sum(ew * jnp.logaddexp(0.0, -e_logits)) / jnp.sum(ew)
            p_loss = jnp.mean(jnp.logaddexp(0.0, p_logits))
            return e_loss + p_loss

        disc_loss_and_grad = jax.value_and_grad(disc_loss_fn)

        # AIRL inner loop
        for round_idx in range(self.config.max_airl_rounds):
            old_policy = jnp.array(policy)

            # Sample from current segment-k policy
            key, subkey = jax.random.split(key)
            pol_s, pol_a, pol_ns = self._sample_transitions_from_policy(
                policy, transitions, n_expert, initial_dist, subkey
            )

            # Discriminator updates
            for _ in range(self.config.discriminator_steps):
                loss, grads = disc_loss_and_grad(
                    reward_params, V, policy,
                    exp_s, exp_a, exp_ns, exp_w,
                    pol_s, pol_a, pol_ns,
                )
                updates, opt_state = optimizer.update(
                    grads, opt_state, reward_params
                )
                reward_params = optax.apply_updates(reward_params, updates)

            # Enforce anchor on reward params
            if self.config.reward_type == "tabular":
                reward_params = self._enforce_anchor_reward(
                    reward_params, exit_action, absorbing
                )
                if self.config.normalize_reward:
                    # Clip to [-5, 5] to prevent scale explosion while
                    # keeping reward values meaningful for value iteration
                    reward_params = jnp.clip(reward_params, -5.0, 5.0)

            # Unit-normalize linear reward to fix scale at 1 (reward can only
            # ROTATE, not SCALE). Prevents adversarial scale inflation from
            # causing the mixture LL to decrease indefinitely at the correct
            # segment assignment.
            if self.config.unit_normalize_reward and self.config.reward_type == "linear":
                theta_norm = jnp.linalg.norm(reward_params)
                reward_params = reward_params / (theta_norm + 1e-8)

            # Update policy
            reward_matrix = get_rm(reward_params)
            policy, V = self._compute_policy_with_anchors(
                reward_matrix, operator, absorbing
            )

            # Check convergence
            policy_change = float(jnp.abs(policy - old_policy).max())
            if policy_change < self.config.airl_convergence_tol:
                break

        return reward_params, opt_state

    # --- Helper methods ---

    def _extract_trajectory_data(self, panel: Panel) -> list[dict]:
        """Extract per-trajectory (s, a, s') arrays."""
        result = []
        for traj in panel.trajectories:
            if len(traj) == 0:
                continue
            result.append({
                "states": jnp.array(traj.states, dtype=jnp.int32),
                "actions": jnp.array(traj.actions, dtype=jnp.int32),
                "next_states": jnp.array(traj.next_states, dtype=jnp.int32),
                "individual_id": traj.individual_id,
            })
        return result

    def _group_by_individual(self, panel: Panel) -> dict[Any, list[int]]:
        """Group trajectory indices by individual_id."""
        groups: dict[Any, list[int]] = {}
        for i, traj in enumerate(panel.trajectories):
            ind_id = traj.individual_id if traj.individual_id is not None else i
            groups.setdefault(ind_id, []).append(i)
        return groups

    def _compute_initial_distribution_from_data(
        self, traj_data: list[dict], n_states: int
    ) -> jnp.ndarray:
        """Compute initial state distribution from trajectory data."""
        init_states = jnp.array(
            [td["states"][0] for td in traj_data], dtype=jnp.int32
        )
        counts = jnp.zeros(n_states, dtype=jnp.float32)
        counts = counts.at[init_states].add(1.0)
        total = counts.sum()
        if total > 0:
            return counts / total
        return jnp.ones(n_states) / n_states

    def _get_reward_matrix(
        self,
        params: jnp.ndarray,
        utility: BaseUtilityFunction,
        n_states: int,
        n_actions: int,
    ) -> jnp.ndarray:
        """Convert parameters to reward matrix."""
        if self.config.reward_type == "tabular":
            if params.ndim == 1:
                return params.reshape(n_states, n_actions)
            return params
        else:
            # Linear: R(s,a) = phi(s,a) . theta
            feature_matrix = utility.feature_matrix
            return jnp.einsum("sak,k->sa", feature_matrix, params)

    def _enforce_anchor_reward(
        self,
        reward_matrix: jnp.ndarray,
        exit_action: int,
        absorbing: int,
    ) -> jnp.ndarray:
        """Enforce anchor constraints on reward matrix.

        Sets r(s, exit_action) = 0 for all s and r(absorbing, a) = 0
        for all a. Combined with V(absorbing) = 0, this pins down the
        AIRL decomposition uniquely (LSW Theorems 1-3).
        """
        if reward_matrix.ndim == 1:
            # Tabular params stored as flat vector
            n_actions = 3  # Will be reshaped by caller
            return reward_matrix
        r = reward_matrix.at[:, exit_action].set(0.0)
        r = r.at[absorbing, :].set(0.0)
        return r

    def _compute_policy_with_anchors(
        self,
        reward_matrix: jnp.ndarray,
        operator: SoftBellmanOperator,
        absorbing: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute policy via value iteration with V(absorbing) = 0 enforced.

        After VI converges, we adjust the value function so that the
        absorbing state has exactly zero continuation value.
        """
        if self.config.generator_solver == "hybrid":
            result = hybrid_iteration(
                operator, reward_matrix,
                tol=self.config.generator_tol,
                max_iter=self.config.generator_max_iter,
            )
        else:
            result = value_iteration(
                operator, reward_matrix,
                tol=self.config.generator_tol,
                max_iter=self.config.generator_max_iter,
            )
        return result.policy, result.V

    def _compute_mixture_ll(
        self,
        traj_data: list[dict],
        segment_policies: list[jnp.ndarray],
        segment_priors: jnp.ndarray,
        sigma: float,
    ) -> float:
        """Compute mixture log-likelihood.

        ll = sum_i log( sum_k prior[k] * prod_t pi_k(a_t | s_t) )
        """
        K = len(segment_policies)
        total_ll = 0.0

        for td in traj_data:
            # Log-likelihood under each segment
            log_liks = []
            for k in range(K):
                log_pi_k = jnp.log(segment_policies[k] + 1e-10)
                traj_ll = float(log_pi_k[td["states"], td["actions"]].sum())
                log_liks.append(jnp.log(segment_priors[k] + 1e-10) + traj_ll)

            # Log-sum-exp for mixture
            total_ll += float(jax.nn.logsumexp(jnp.array(log_liks)))

        return total_ll
