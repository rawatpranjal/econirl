"""f-IRL: Inverse Reinforcement Learning via State Marginal Matching.

Recovers reward functions by matching the state-marginal distribution of
the policy to the expert's empirical state-marginal, using f-divergence
minimization instead of feature expectation matching.

Algorithm:
    1. Compute expert state-action marginal from demonstrations
    2. Initialize tabular reward R(s, a)
    3. For each iteration:
       a. Solve MDP under current R to get policy pi
       b. Compute policy state-action marginal via forward propagation
       c. Compute f-divergence gradient between expert and policy marginals
       d. Update R in the divergence-gradient direction
    4. Return R and induced policy

Supported f-divergences:
    - fkl (or "kl"): forward KL D_KL(p_expert || p_policy),
        gradient = log(p_expert / p_policy)
    - rkl: reverse KL D_KL(p_policy || p_expert) (mode-seeking),
        gradient = log(p_policy / p_expert)
    - js: Jensen-Shannon divergence, symmetric mixture-based form,
        gradient = log(p_expert / mean) - log(p_policy / mean) where
        mean = (p_expert + p_policy) / 2
    - chi2: chi-squared (econirl extension), gradient = (p_expert / p_policy) - 1
    - tv: total variation (econirl extension), gradient = sign(p_expert - p_policy)

Reference:
    Ni, T., Sikchi, H., Wang, Y., Gupta, T., Lee, L., & Eysenbach, B. (2022).
    "f-IRL: Inverse Reinforcement Learning via State Marginal Matching."
    CoRL.
"""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import BaseUtilityFunction, UtilityFunction


class FIRLEstimator(BaseEstimator):
    """f-IRL estimator via state-marginal matching.

    Recovers a tabular reward function R(s,a) by minimizing the
    f-divergence between the expert's state-action marginal and the
    policy's state-action marginal. This avoids the feature-matching
    assumption of MaxEnt IRL.

    Attributes:
        f_divergence: Which f-divergence to use ("kl", "chi2", "tv").
        lr: Learning rate for reward updates.
        max_iter: Maximum number of gradient iterations.
        inner_tol: Convergence tolerance for MDP solver.
        inner_max_iter: Maximum iterations for MDP solver.
        horizon: Horizon for state visitation computation.

    Example:
        >>> estimator = FIRLEstimator(f_divergence="kl", lr=0.5)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        f_divergence: Literal["kl", "fkl", "rkl", "js", "chi2", "tv"] = "fkl",
        lr: float = 0.5,
        max_iter: int = 500,
        inner_tol: float = 1e-8,
        inner_max_iter: int = 5000,
        horizon: int = 100,
        reward_clip: float = 10.0,
        compute_se: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=verbose,
        )
        # "kl" is a back-compat alias for "fkl" (forward KL) per Ni et al. 2022.
        self._f_divergence = "fkl" if f_divergence == "kl" else f_divergence
        self._lr = lr
        self._max_iter = max_iter
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._horizon = horizon
        self._reward_clip = reward_clip

    @property
    def name(self) -> str:
        return f"f-IRL ({self._f_divergence}, Ni et al. 2022)"

    def _compute_expert_marginal(
        self,
        panel: Panel,
        n_states: int,
        n_actions: int,
    ) -> jnp.ndarray:
        """Compute empirical state-action marginal from demonstrations.

        Returns:
            State-action marginal, shape (n_states, n_actions), sums to 1.
        """
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        idx = (all_states * n_actions + all_actions).astype(jnp.int32)
        counts = jnp.zeros(n_states * n_actions)
        counts = counts.at[idx].add(1.0)
        counts = counts.reshape(n_states, n_actions)
        total = counts.sum()
        return counts / jnp.maximum(total, 1.0)

    def _compute_policy_marginal(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        panel: Panel,
    ) -> jnp.ndarray:
        """Compute state-action marginal under policy via forward propagation.

        Returns:
            State-action marginal, shape (n_states, n_actions), sums to 1.
        """
        n_states = problem.num_states
        beta = problem.discount_factor

        # Initial state distribution from data
        init_counts = jnp.zeros(n_states)
        init_states = jnp.array(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        init_counts = init_counts.at[init_states].add(1.0)
        mu = init_counts / jnp.maximum(init_counts.sum(), 1.0)

        # Forward propagation of state visitation
        state_vis = mu
        P_pi = jnp.einsum("sa,ast->st", policy, transitions)

        for t in range(1, self._horizon):
            mu = mu @ P_pi
            state_vis += (beta ** t) * mu

        state_vis = state_vis / state_vis.sum()

        # State-action marginal: d(s) * pi(a|s)
        sa_marginal = state_vis[:, None] * policy
        return sa_marginal

    def _f_divergence_gradient(
        self,
        p_expert: jnp.ndarray,
        p_policy: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute gradient of f-divergence w.r.t. reward.

        The reward update direction is proportional to the f-divergence
        gradient evaluated at the density ratio.

        Args:
            p_expert: Expert marginal, shape (n_states, n_actions).
            p_policy: Policy marginal, shape (n_states, n_actions).

        Returns:
            Gradient direction, shape (n_states, n_actions).
        """
        eps = 1e-10
        p_policy_safe = jnp.clip(p_policy, min=eps)
        p_expert_safe = jnp.clip(p_expert, min=eps)

        if self._f_divergence == "fkl":
            # Forward KL D_KL(p_E || p_pi). Mass-covering: upweights states
            # where expert has support but policy does not.
            return jnp.log(p_expert_safe / p_policy_safe)
        elif self._f_divergence == "rkl":
            # Reverse KL D_KL(p_pi || p_E). Mode-seeking: penalizes states
            # where the policy puts mass that the expert does not.
            return jnp.log(p_policy_safe / p_expert_safe)
        elif self._f_divergence == "js":
            # Jensen-Shannon symmetric divergence with mixture
            # M = (p_E + p_pi) / 2.
            mean = 0.5 * (p_expert_safe + p_policy_safe)
            return jnp.log(p_expert_safe / mean) - jnp.log(p_policy_safe / mean)
        elif self._f_divergence == "chi2":
            return (p_expert_safe / p_policy_safe) - 1.0
        elif self._f_divergence == "tv":
            return jnp.sign(p_expert - p_policy)
        else:
            raise ValueError(f"Unknown f-divergence: {self._f_divergence}")

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run f-IRL optimization."""
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        operator = SoftBellmanOperator(problem, transitions)

        # Expert marginal
        expert_marginal = self._compute_expert_marginal(panel, n_states, n_actions)

        # Initialize tabular reward
        reward = jnp.zeros((n_states, n_actions))

        best_ll = float("-inf")
        best_policy = None
        best_V = None
        best_reward = None

        self._log(f"f-IRL ({self._f_divergence}): {self._max_iter} iterations")

        from tqdm import tqdm
        pbar = tqdm(
            range(self._max_iter),
            desc=f"f-IRL ({self._f_divergence})",
            disable=not self._verbose,
            leave=True,
        )
        for it in pbar:
            # Solve MDP under current reward
            solver_result = value_iteration(
                operator, reward,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
            )
            policy = solver_result.policy

            # Compute policy marginal
            policy_marginal = self._compute_policy_marginal(
                policy, transitions, problem, panel,
            )

            # Compute divergence gradient and update reward
            grad = self._f_divergence_gradient(expert_marginal, policy_marginal)
            reward = reward + self._lr * grad
            reward = jnp.clip(reward, -self._reward_clip, self._reward_clip)

            # Track best policy by log-likelihood
            log_probs = operator.compute_log_choice_probabilities(
                reward, solver_result.V,
            )
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            ll = log_probs[all_states, all_actions].sum().item()

            if ll > best_ll:
                best_ll = ll
                best_policy = jnp.array(policy)
                best_V = jnp.array(solver_result.V)
                best_reward = jnp.array(reward)

            div = jnp.abs(expert_marginal - policy_marginal).sum().item()
            r_range = float(jnp.max(reward) - jnp.min(reward))
            pbar.set_postfix({
                "LL": f"{ll:.2f}",
                "best": f"{best_ll:.2f}",
                "div": f"{div:.4f}",
                "R_rng": f"{r_range:.2f}",
            })

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=best_reward.flatten(),
            log_likelihood=best_ll,
            value_function=best_V,
            policy=best_policy,
            hessian=None,
            converged=True,
            num_iterations=self._max_iter,
            message=f"f-IRL ({self._f_divergence}): {self._max_iter} iterations",
            optimization_time=elapsed,
            metadata={
                "reward_matrix": best_reward,
                "expert_marginal": expert_marginal,
                "f_divergence": self._f_divergence,
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
        """Estimate via f-IRL.

        Overrides base to handle tabular reward output.
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
            + n_params * jnp.log(jnp.array(n_obs)).item(),
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
