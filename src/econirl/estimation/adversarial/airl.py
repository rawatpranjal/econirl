"""Adversarial Inverse Reinforcement Learning (AIRL) for tabular MDPs.

This module implements AIRL (Fu et al. 2018) adapted for discrete choice models.
AIRL recovers a reward function that is robust to changes in dynamics by using
a specific discriminator structure that disentangles reward from shaping.

Algorithm:
    1. Initialize reward function r(s,a) and value function V(s)
    2. Repeat:
       a) Compute discriminator: D(s,a,s') = exp(f) / (exp(f) + pi(a|s))
          where f(s,a,s') = r(s,a) + gamma*V(s') - V(s)
       b) Update discriminator to classify expert vs policy
       c) Update policy to maximize discriminator reward
       d) Update value function estimate

Reference:
    Fu, J., Luo, K., & Levine, S. (2018). "Learning robust rewards with
    adversarial inverse reinforcement learning." ICLR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class AIRLConfig:
    """Configuration for tabular AIRL.

    Attributes:
        reward_type: Parameterization of reward ("tabular" or "linear")
        reward_lr: Learning rate for reward updates
        discriminator_steps: Discriminator updates per round
        generator_solver: Inner solver for policy
        generator_tol: Tolerance for value iteration
        generator_max_iter: Max iterations for value iteration
        max_rounds: Maximum training rounds
        use_shaping: Whether to use potential shaping f = r + gamma*V(s') - V(s)
        shaping_coef: Coefficient for shaping term (typically gamma)
        convergence_tol: Tolerance for policy convergence
        compute_se: Whether to compute standard errors
        se_method: Method for standard errors
        n_bootstrap: Number of bootstrap samples
        verbose: Whether to print progress
    """

    reward_type: Literal["tabular", "linear"] = "tabular"
    reward_lr: float = 0.01
    discriminator_steps: int = 5
    generator_solver: Literal["value", "hybrid"] = "hybrid"
    generator_tol: float = 1e-8
    generator_max_iter: int = 5000
    max_rounds: int = 100
    use_shaping: bool = True
    shaping_coef: float | None = None  # If None, uses discount_factor
    convergence_tol: float = 1e-4
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic"] = "bootstrap"
    n_bootstrap: int = 100
    verbose: bool = False


class AIRLEstimator(BaseEstimator):
    """Adversarial Inverse Reinforcement Learning for tabular MDPs.

    AIRL learns a disentangled reward function that is robust to changes
    in dynamics. The key insight is using a discriminator of the form:

        D(s,a,s') = exp(f) / (exp(f) + pi(a|s))

    where f(s,a,s') = r(s,a) + gamma*V(s') - V(s).

    This structure allows recovery of the reward r(s,a) independent of
    the shaping term V.

    Parameters
    ----------
    config : AIRLConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Examples
    --------
    >>> from econirl.estimation.adversarial import AIRLEstimator, AIRLConfig
    >>> config = AIRLConfig(max_rounds=100, verbose=True)
    >>> estimator = AIRLEstimator(config=config)
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    """

    def __init__(
        self,
        config: AIRLConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = AIRLConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method=config.se_method if config.compute_se else "asymptotic",
            compute_hessian=False,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "AIRL (Fu et al. 2018)"

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate reward function using AIRL.

        Overrides base class to handle reward parameters properly.

        Args:
            panel: Expert demonstrations
            utility: Utility/reward function specification
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

        # Generate parameter names
        if self.config.reward_type == "linear":
            param_names = utility.parameter_names
        else:
            # Tabular reward: one parameter per (state, action) pair
            param_names = [
                f"R({s},{a})"
                for s in range(problem.num_states)
                for a in range(problem.num_actions)
            ]

        # Create standard errors (NaN for adversarial methods)
        standard_errors = torch.full_like(result.parameters, float("nan"))

        # Goodness of fit
        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * torch.log(torch.tensor(n_obs)).item(),
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

    def _sample_transitions_from_panel(
        self,
        panel: Panel,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (s, a, s') transitions from expert demonstrations.

        Returns:
            Tuple of (states, actions, next_states) tensors
        """
        return (
            panel.get_all_states(),
            panel.get_all_actions(),
            panel.get_all_next_states(),
        )

    def _sample_transitions_from_policy(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        n_samples: int,
        initial_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (s, a, s') transitions from current policy.

        Returns:
            Tuple of (states, actions, next_states) tensors
        """
        n_states, n_actions = policy.shape
        states = []
        actions = []
        next_states_list = []

        state = torch.multinomial(initial_dist, 1).item()

        for _ in range(n_samples):
            action = torch.multinomial(policy[state], 1).item()
            next_state_dist = transitions[action, state, :]
            next_state = torch.multinomial(next_state_dist, 1).item()

            states.append(state)
            actions.append(action)
            next_states_list.append(next_state)

            state = next_state

        return (
            torch.tensor(states, dtype=torch.long),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(next_states_list, dtype=torch.long),
        )

    def _compute_airl_logits(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        reward_matrix: torch.Tensor,
        V: torch.Tensor,
        policy: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Compute AIRL discriminator logits.

        D(s,a,s') = sigmoid(f - log pi(a|s))
        where f = r(s,a) + gamma*V(s') - V(s)

        Returns logits = f - log pi(a|s)
        """
        # f(s,a,s') = r(s,a) + gamma*V(s') - V(s)
        r_sa = reward_matrix[states, actions]
        if self.config.use_shaping:
            shaping_coef = (
                self.config.shaping_coef if self.config.shaping_coef else gamma
            )
            f = r_sa + shaping_coef * V[next_states] - V[states]
        else:
            f = r_sa

        # log pi(a|s)
        log_pi = torch.log(policy[states, actions] + 1e-10)

        # AIRL logit
        return f - log_pi

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> torch.Tensor:
        """Compute initial state distribution from data."""
        counts = torch.zeros(n_states, dtype=torch.float32)
        init_states = torch.tensor(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=torch.long,
        )
        counts.scatter_add_(0, init_states, torch.ones_like(init_states, dtype=torch.float32))

        if counts.sum() > 0:
            return counts / counts.sum()
        return torch.ones(n_states) / n_states

    def _compute_policy(
        self,
        reward_matrix: torch.Tensor,
        operator: SoftBellmanOperator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal policy given reward matrix."""
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

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run AIRL optimization.

        Args:
            panel: Expert demonstrations
            utility: Reward function specification
            problem: Problem specification
            transitions: Transition matrices
            initial_params: Initial reward parameters (optional)

        Returns:
            EstimationResult with learned reward and policy
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        gamma = problem.discount_factor
        operator = SoftBellmanOperator(problem, transitions)

        # Initialize reward matrix
        if self.config.reward_type == "linear":
            if isinstance(utility, ActionDependentReward):
                feature_matrix = utility.feature_matrix
                n_features = feature_matrix.shape[2]
            elif isinstance(utility, LinearReward):
                sf = utility.state_features
                feature_matrix = sf.unsqueeze(1).expand(-1, n_actions, -1).clone()
                n_features = sf.shape[1]
            else:
                raise TypeError(f"Unsupported utility type: {type(utility)}")

            if initial_params is not None:
                reward_weights = initial_params.clone()
            else:
                reward_weights = torch.zeros(n_features)

            def get_reward_matrix():
                return torch.einsum("sak,k->sa", feature_matrix, reward_weights)

        else:
            # Tabular reward
            reward_matrix = torch.zeros(n_states, n_actions)
            reward_weights = None

            def get_reward_matrix():
                return reward_matrix

        # Initial state distribution
        initial_dist = self._compute_initial_distribution(panel, n_states)

        # Sample expert transitions once
        expert_states, expert_actions, expert_next_states = (
            self._sample_transitions_from_panel(panel)
        )
        n_expert = len(expert_states)

        # Initialize policy
        policy = torch.ones(n_states, n_actions) / n_actions
        V = torch.zeros(n_states)

        # Training metrics
        disc_losses = []
        policy_changes = []
        converged = False
        round_idx = 0

        pbar = tqdm(
            range(self.config.max_rounds),
            desc="AIRL",
            disable=not self.config.verbose,
        )

        for round_idx in pbar:
            old_policy = policy.clone()
            current_reward = get_reward_matrix()

            # Sample from current policy
            policy_states, policy_actions, policy_next_states = (
                self._sample_transitions_from_policy(
                    policy, transitions, n_expert, initial_dist
                )
            )

            # Update reward (discriminator) with gradient
            disc_loss = 0.0
            for _ in range(self.config.discriminator_steps):
                # Expert: want D to be high (label = 1)
                expert_logits = self._compute_airl_logits(
                    expert_states,
                    expert_actions,
                    expert_next_states,
                    current_reward,
                    V,
                    policy,
                    gamma,
                )

                # Policy: want D to be low (label = 0)
                policy_logits = self._compute_airl_logits(
                    policy_states,
                    policy_actions,
                    policy_next_states,
                    current_reward,
                    V,
                    policy,
                    gamma,
                )

                # BCE loss
                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_logits, torch.ones_like(expert_logits)
                )
                policy_loss = F.binary_cross_entropy_with_logits(
                    policy_logits, torch.zeros_like(policy_logits)
                )
                disc_loss = (expert_loss + policy_loss).item()

                # Gradient update for reward
                # d(logit)/d(r) = 1 for the (s,a) pair
                # d(BCE)/d(logit) = sigmoid(logit) - label
                expert_grad = torch.sigmoid(expert_logits) - 1.0
                policy_grad = torch.sigmoid(policy_logits) - 0.0

                if self.config.reward_type == "linear":
                    # Gradient w.r.t. weights
                    expert_features = feature_matrix[expert_states, expert_actions, :]
                    policy_features = feature_matrix[policy_states, policy_actions, :]
                    grad = (expert_grad.unsqueeze(1) * expert_features).mean(dim=0)
                    grad += (policy_grad.unsqueeze(1) * policy_features).mean(dim=0)
                    reward_weights = reward_weights - self.config.reward_lr * grad
                else:
                    # Direct update to reward matrix
                    for i, (s, a) in enumerate(zip(expert_states, expert_actions)):
                        reward_matrix[s, a] -= self.config.reward_lr * expert_grad[i]
                    for i, (s, a) in enumerate(zip(policy_states, policy_actions)):
                        reward_matrix[s, a] -= self.config.reward_lr * policy_grad[i]

                current_reward = get_reward_matrix()

            disc_losses.append(disc_loss)

            # Update policy via soft value iteration
            policy, V = self._compute_policy(current_reward, operator)

            # Check convergence
            policy_change = torch.abs(policy - old_policy).max().item()
            policy_changes.append(policy_change)

            pbar.set_postfix(
                {
                    "disc_loss": f"{disc_loss:.4f}",
                    "policy_change": f"{policy_change:.4f}",
                }
            )

            if policy_change < self.config.convergence_tol:
                converged = True
                break

        pbar.close()

        # Final values
        final_reward = get_reward_matrix()

        # Compute log-likelihood
        log_probs = operator.compute_log_choice_probabilities(final_reward, V)
        ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()

        # Extract parameters
        if self.config.reward_type == "linear":
            parameters = reward_weights.clone()
        else:
            parameters = reward_matrix.flatten()

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
                "reward_type": self.config.reward_type,
                "use_shaping": self.config.use_shaping,
                "final_disc_loss": disc_losses[-1] if disc_losses else None,
                "disc_losses": disc_losses,
                "policy_changes": policy_changes,
                "reward_matrix": final_reward.tolist(),
            },
        )
