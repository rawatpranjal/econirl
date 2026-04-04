"""IQ-Learn: Inverse soft-Q Learning for Imitation.

This module implements IQ-Learn (Garg et al. 2021) adapted for tabular DDC models.
IQ-Learn learns a single soft Q-function that implicitly represents both reward
and policy, avoiding adversarial training entirely.

Algorithm:
    1. Parameterize Q(s,a) as tabular or linear in features
    2. Optimize the IQ-Learn objective (concave in Q):
       - Chi-squared (offline): min_Q -E_rho[Q(s,a)-V*(s)] + (1/4a)E_rho[td^2]
       - Simple (TV distance): max_Q E_rho[td] - (1-gamma)E_p0[V*(s0)]
    3. Extract policy: pi(a|s) = softmax(Q(s,a)/sigma)
    4. Recover reward: r(s,a) = Q(s,a) - gamma * E[V*(s')]

Reference:
    Garg, D., Chakraborty, S., Cundy, C., Song, J., & Ermon, S. (2021).
    "IQ-Learn: Inverse soft-Q Learning for Imitation." NeurIPS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class IQLearnConfig:
    """Configuration for IQ-Learn estimation.

    Attributes:
        q_type: Parameterization of Q-function ("tabular" or "linear")
        divergence: Statistical divergence for the objective
        alpha: Regularization strength for chi-squared divergence
        optimizer: Optimization method
        learning_rate: Learning rate for Adam optimizer
        max_iter: Maximum optimization iterations
        convergence_tol: Gradient norm convergence tolerance
        verbose: Whether to print progress
    """

    q_type: Literal["tabular", "linear"] = "tabular"
    divergence: Literal["chi2", "simple"] = "chi2"
    alpha: float = 1.0
    optimizer: Literal["L-BFGS-B", "adam"] = "L-BFGS-B"
    learning_rate: float = 0.01
    max_iter: int = 500
    convergence_tol: float = 1e-6
    verbose: bool = False


class IQLearnEstimator(BaseEstimator):
    """Inverse soft-Q Learning for tabular MDPs.

    IQ-Learn learns a soft Q-function that implicitly defines both the optimal
    policy and reward function. The key insight is that the IRL min-max problem
    over (reward, policy) collapses to a concave maximization over Q alone,
    since the optimal policy is deterministically given by softmax(Q).

    Parameters
    ----------
    config : IQLearnConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Examples
    --------
    >>> from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
    >>> config = IQLearnConfig(q_type="linear", divergence="chi2")
    >>> estimator = IQLearnEstimator(config=config)
    >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        config: IQLearnConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = IQLearnConfig(**kwargs)
        else:
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
        return "IQ-Learn (Garg et al. 2021)"

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: np.ndarray | jnp.ndarray,
        initial_params: np.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate reward function using IQ-Learn.

        Overrides base class to handle Q-function parameter naming.
        """
        start_time = time.time()

        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        if len(result.parameters) == utility.num_parameters:
            param_names = utility.parameter_names
        else:
            param_names = [
                f"R({s},{a})"
                for s in range(problem.num_states)
                for a in range(problem.num_actions)
            ]

        standard_errors = jnp.full_like(result.parameters, float("nan"))

        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * np.log(n_obs),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, jnp.asarray(result.policy)
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

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute empirical initial state distribution from data."""
        counts = np.zeros(n_states, dtype=np.float64)
        for traj in panel.trajectories:
            if len(traj) > 0:
                s0 = int(np.asarray(traj.states[0]))
                counts[s0] += 1
        total = counts.sum()
        if total > 0:
            counts = counts / total
        else:
            counts = np.ones(n_states, dtype=np.float64) / n_states
        return jnp.array(counts)

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: np.ndarray | jnp.ndarray,
        initial_params: np.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run IQ-Learn optimization.

        Learns a soft Q-function by optimizing the IQ-Learn objective,
        then extracts policy and reward via the inverse Bellman operator.
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        gamma = problem.discount_factor
        sigma = problem.scale_parameter
        alpha = self.config.alpha

        # Extract expert (s, a, s') from panel
        expert_states = np.asarray(panel.get_all_states(), dtype=np.int64)
        expert_actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
        expert_next_states = np.asarray(panel.get_all_next_states(), dtype=np.int64)

        # Convert to JAX for use inside objective
        expert_states_jax = jnp.array(expert_states)
        expert_actions_jax = jnp.array(expert_actions)

        # Initial state distribution (needed for simple divergence)
        initial_dist = self._compute_initial_distribution(panel, n_states)

        # Transitions as float64 for numerical precision
        trans_f64 = jnp.asarray(transitions, dtype=jnp.float64)

        # Setup Q parameterization
        if self.config.q_type == "linear":
            if isinstance(utility, ActionDependentReward):
                fm = np.asarray(utility.feature_matrix, dtype=np.float64)
                feature_matrix = jnp.array(fm)
                n_params = feature_matrix.shape[2]
            elif isinstance(utility, LinearReward):
                sf = np.asarray(utility.state_features, dtype=np.float64)
                sf_jax = jnp.array(sf)
                feature_matrix = jnp.broadcast_to(
                    sf_jax[:, None, :], (n_states, n_actions, sf_jax.shape[1])
                )
                n_params = sf_jax.shape[1]
            else:
                raise TypeError(f"Unsupported utility type for linear q_type: {type(utility)}")

            if initial_params is not None:
                theta_init = np.asarray(initial_params, dtype=np.float64)
            else:
                theta_init = np.zeros(n_params)
        else:
            # Tabular: free Q(s,a) matrix
            feature_matrix = None
            n_params = n_states * n_actions
            if initial_params is not None:
                theta_init = np.asarray(initial_params, dtype=np.float64)
            else:
                theta_init = np.zeros(n_params)

        divergence = self.config.divergence

        def objective(theta):
            """Compute IQ-Learn objective (JAX, differentiable)."""
            # Build Q table
            if feature_matrix is not None:
                Q = jnp.einsum("sak,k->sa", feature_matrix, theta)
            else:
                Q = theta.reshape(n_states, n_actions)

            # V*(s) = sigma * logsumexp(Q(s,:) / sigma)
            V_star = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)

            # E_{s'~P(.|s,a)}[V*(s')] = sum_{s'} P(s'|s,a) V*(s')
            # transitions shape: (A, S, S') -> einsum to get (A, S)
            EV = jnp.einsum("ast,t->as", trans_f64, V_star).T  # (S, A)

            # Temporal difference: Q(s,a) - gamma * E[V*(s')]
            td = Q - gamma * EV

            # Expert terms
            Q_expert = Q[expert_states_jax, expert_actions_jax]
            V_expert = V_star[expert_states_jax]

            if divergence == "chi2":
                td_expert = td[expert_states_jax, expert_actions_jax]
                loss = -(Q_expert - V_expert).mean() + (1.0 / (4 * alpha)) * (td_expert**2).mean()
            else:
                td_expert = td[expert_states_jax, expert_actions_jax]
                loss = -td_expert.mean() + (1 - gamma) * jnp.dot(initial_dist, V_star)

            return loss

        obj_and_grad = jax.value_and_grad(objective)

        def objective_and_gradient(theta_np):
            """Wrapper for scipy: numpy in, numpy out."""
            theta_jax = jnp.array(theta_np)
            loss, grad = obj_and_grad(theta_jax)
            return float(loss), np.asarray(grad, dtype=np.float64)

        # Optimize
        if self.config.optimizer == "L-BFGS-B":
            from tqdm import tqdm
            _iq_pbar = tqdm(
                total=self.config.max_iter,
                desc="IQ-Learn L-BFGS-B",
                disable=not self.config.verbose,
                leave=True,
            )

            def _iq_callback(xk):
                _iq_pbar.update(1)

            result_scipy = optimize.minimize(
                objective_and_gradient,
                theta_init,
                method="L-BFGS-B",
                jac=True,
                callback=_iq_callback,
                options={
                    "maxiter": self.config.max_iter,
                    "gtol": self.config.convergence_tol,
                },
            )
            _iq_pbar.close()
            theta_opt = jnp.array(result_scipy.x)
            converged = result_scipy.success
            num_iterations = result_scipy.nit
            num_fevals = result_scipy.nfev
            message = result_scipy.message if isinstance(result_scipy.message, str) else result_scipy.message.decode()
            final_obj = result_scipy.fun
        else:
            # Adam optimizer (manual implementation)
            theta_np = theta_init.copy()
            m = np.zeros_like(theta_np)
            v = np.zeros_like(theta_np)
            lr = self.config.learning_rate
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            loss_history = []
            grad_norm = float("inf")

            from tqdm import tqdm
            pbar = tqdm(
                range(1, self.config.max_iter + 1),
                desc="IQ-Learn Adam",
                disable=not self.config.verbose,
                leave=True,
            )
            for t in pbar:
                obj, grad_np = objective_and_gradient(theta_np)
                loss_history.append(obj)
                pbar.set_postfix({"loss": f"{obj:.4f}", "|g|": f"{grad_norm:.2e}"})

                m = beta1 * m + (1 - beta1) * grad_np
                v = beta2 * v + (1 - beta2) * grad_np**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                theta_np = theta_np - lr * m_hat / (np.sqrt(v_hat) + eps)

                grad_norm = float(np.linalg.norm(grad_np))
                if grad_norm < self.config.convergence_tol:
                    break

            theta_opt = jnp.array(theta_np)
            converged = grad_norm < self.config.convergence_tol
            num_iterations = t
            num_fevals = t
            message = "Converged" if converged else "Max iterations reached"
            final_obj = loss_history[-1] if loss_history else float("nan")

        # Extract results from optimal Q
        if feature_matrix is not None:
            Q_table = jnp.einsum("sak,k->sa", feature_matrix, theta_opt).astype(jnp.float32)
        else:
            Q_table = theta_opt.reshape(n_states, n_actions).astype(jnp.float32)

        # Policy: softmax(Q/sigma)
        policy = jax.nn.softmax(Q_table / sigma, axis=1)

        # Value function: V*(s) = sigma * logsumexp(Q/sigma)
        V = sigma * jax.scipy.special.logsumexp(Q_table / sigma, axis=1)

        # Reward via inverse Bellman: r(s,a) = Q(s,a) - gamma * E[V*(s')]
        transitions_f32 = jnp.asarray(transitions, dtype=jnp.float32)
        EV = jnp.einsum("ast,t->as", transitions_f32, V).T
        reward_table = Q_table - gamma * EV

        # Log-likelihood
        log_probs = jax.nn.log_softmax(Q_table / sigma, axis=1)
        ll = float(log_probs[expert_states_jax, expert_actions_jax].sum())

        # Project reward onto feature space for structural parameters
        reward_params = None
        feat = None
        if hasattr(utility, 'feature_matrix'):
            feat = jnp.asarray(utility.feature_matrix, dtype=jnp.float32)
        elif isinstance(utility, LinearReward) and hasattr(utility, 'state_features'):
            sf = jnp.asarray(utility.state_features, dtype=jnp.float32)
            feat = jnp.broadcast_to(sf[:, None, :], (n_states, n_actions, sf.shape[1]))

        if feat is not None:
            Phi = feat.reshape(-1, feat.shape[2])  # (S*A, K)
            r_flat = reward_table.flatten()          # (S*A,)
            # Add constant column to absorb additive offset from shaping
            Phi_aug = jnp.concatenate([Phi, jnp.ones((Phi.shape[0], 1))], axis=1)
            params_aug = jnp.linalg.lstsq(Phi_aug, r_flat, rcond=None)[0]
            reward_params = params_aug[:-1]  # Drop constant term

        # Parameters to return
        if self.config.q_type == "linear":
            parameters = theta_opt.astype(jnp.float32)
        elif reward_params is not None:
            parameters = reward_params.astype(jnp.float32)
        else:
            parameters = reward_table.flatten().astype(jnp.float32)

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=parameters,
            log_likelihood=ll,
            value_function=V,
            policy=policy,
            hessian=None,
            converged=converged,
            num_iterations=num_iterations,
            num_function_evals=num_fevals,
            message=message,
            optimization_time=optimization_time,
            metadata={
                "q_type": self.config.q_type,
                "divergence": self.config.divergence,
                "alpha": self.config.alpha,
                "q_table": np.asarray(Q_table).tolist(),
                "reward_table": np.asarray(reward_table).tolist(),
                "reward_params": np.asarray(reward_params).tolist() if reward_params is not None else None,
                "final_objective": final_obj,
            },
        )
