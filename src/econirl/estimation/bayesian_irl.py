"""Bayesian Inverse Reinforcement Learning (BIRL) estimator.

Implements MCMC-based reward inference via Metropolis-Hastings sampling
over reward parameters. Each proposal is evaluated by solving the MDP
and computing the likelihood of observed behavior under the induced policy.

The posterior mean serves as the point estimate and posterior standard
deviations provide uncertainty quantification without needing asymptotic
approximations.

Algorithm:
    1. Initialize theta ~ Prior
    2. For each MCMC iteration:
       a. Propose theta' ~ N(theta, proposal_sigma^2 I)
       b. Solve MDP under theta' to get policy pi'
       c. Compute log-likelihood of data under pi'
       d. Accept/reject via Metropolis-Hastings ratio
    3. Return posterior mean (point estimate) and posterior std (SEs)

Reference:
    Ramachandran, D. & Amir, E. (2007).
    "Bayesian Inverse Reinforcement Learning."
    IJCAI, pp. 2586-2591.
"""

from __future__ import annotations

import time

import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import BaseUtilityFunction, UtilityFunction


class BayesianIRLEstimator(BaseEstimator):
    """Bayesian Inverse Reinforcement Learning estimator.

    Uses Metropolis-Hastings MCMC to sample from the posterior distribution
    over reward parameters given observed behavior. The likelihood is derived
    from the softmax (logit) policy induced by solving the MDP at each
    proposed parameter vector.

    The posterior mean provides a point estimate and the posterior standard
    deviation provides standard errors without asymptotic approximations.

    Attributes:
        n_samples: Total number of MCMC samples.
        burnin: Number of initial samples to discard.
        proposal_sigma: Standard deviation of Gaussian proposal distribution.
        prior_sigma: Standard deviation of Gaussian prior on parameters.
        confidence: Confidence parameter (inverse temperature) for likelihood.

    Example:
        >>> estimator = BayesianIRLEstimator(n_samples=2000, burnin=500)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.parameters)  # Posterior mean
    """

    def __init__(
        self,
        n_samples: int = 2000,
        burnin: int = 500,
        proposal_sigma: float = 0.1,
        prior_sigma: float = 5.0,
        confidence: float = 1.0,
        inner_tol: float = 1e-8,
        inner_max_iter: int = 5000,
        compute_se: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=verbose,
        )
        self._n_samples = n_samples
        self._burnin = burnin
        self._proposal_sigma = proposal_sigma
        self._prior_sigma = prior_sigma
        self._confidence = confidence
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._compute_se = compute_se

    @property
    def name(self) -> str:
        return "Bayesian IRL (Ramachandran & Amir 2007)"

    def _log_prior(self, params: torch.Tensor) -> float:
        """Gaussian prior: log p(theta) = -||theta||^2 / (2 * sigma^2)."""
        return (-0.5 * (params ** 2).sum() / (self._prior_sigma ** 2)).item()

    def _log_likelihood(
        self,
        params: torch.Tensor,
        panel: Panel,
        utility: BaseUtilityFunction,
        operator: SoftBellmanOperator,
    ) -> float:
        """Compute log-likelihood of data under policy induced by params.

        Solves the MDP for the given parameters and computes the log
        probability of each observed (state, action) pair under the
        resulting softmax policy.
        """
        reward_matrix = utility.compute(params)
        solver_result = value_iteration(
            operator, reward_matrix,
            tol=self._inner_tol,
            max_iter=self._inner_max_iter,
        )

        log_probs = operator.compute_log_choice_probabilities(
            reward_matrix, solver_result.V
        )

        ll = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ll += log_probs[s, a].item()

        return self._confidence * ll

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run Metropolis-Hastings MCMC sampling."""
        start_time = time.time()

        operator = SoftBellmanOperator(problem, transitions)
        n_params = utility.num_parameters

        # Initialize at prior mean (zeros) or provided initial params
        if initial_params is None:
            current_params = utility.get_initial_parameters()
        else:
            current_params = initial_params.clone()

        current_log_post = (
            self._log_likelihood(current_params, panel, utility, operator)
            + self._log_prior(current_params)
        )

        # Storage for samples
        samples = torch.zeros(self._n_samples, n_params)
        log_posteriors = torch.zeros(self._n_samples)
        n_accepted = 0

        self._log(f"Starting MCMC: {self._n_samples} samples, burnin={self._burnin}")

        for i in range(self._n_samples):
            # Propose new parameters
            proposal = current_params + self._proposal_sigma * torch.randn(n_params)

            # Compute log-posterior of proposal
            proposal_log_post = (
                self._log_likelihood(proposal, panel, utility, operator)
                + self._log_prior(proposal)
            )

            # Metropolis-Hastings acceptance (symmetric proposal)
            log_alpha = proposal_log_post - current_log_post
            if torch.log(torch.rand(1)).item() < log_alpha:
                current_params = proposal
                current_log_post = proposal_log_post
                n_accepted += 1

            samples[i] = current_params
            log_posteriors[i] = current_log_post

            if self._verbose and (i + 1) % 200 == 0:
                accept_rate = n_accepted / (i + 1)
                self._log(
                    f"  Sample {i+1}/{self._n_samples}: "
                    f"accept_rate={accept_rate:.2f}, "
                    f"log_post={current_log_post:.2f}"
                )

        # Discard burn-in
        post_burnin = samples[self._burnin:]
        accept_rate = n_accepted / self._n_samples

        # Posterior mean as point estimate
        posterior_mean = post_burnin.mean(dim=0)
        posterior_std = post_burnin.std(dim=0)

        self._log(f"MCMC complete: accept_rate={accept_rate:.3f}")
        self._log(f"Posterior mean: {posterior_mean.numpy()}")

        # Compute final policy from posterior mean
        reward_matrix = utility.compute(posterior_mean)
        solver_result = value_iteration(
            operator, reward_matrix,
            tol=self._inner_tol,
            max_iter=self._inner_max_iter,
        )

        # Final log-likelihood at posterior mean
        log_probs = operator.compute_log_choice_probabilities(
            reward_matrix, solver_result.V
        )
        ll = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ll += log_probs[s, a].item()

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=posterior_mean,
            log_likelihood=ll,
            value_function=solver_result.V,
            policy=solver_result.policy,
            hessian=None,
            converged=True,
            num_iterations=self._n_samples,
            message=f"MCMC: accept_rate={accept_rate:.3f}",
            optimization_time=elapsed,
            metadata={
                "posterior_std": posterior_std,
                "accept_rate": accept_rate,
                "n_samples": self._n_samples,
                "burnin": self._burnin,
                "samples": post_burnin,
            },
        )

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate reward parameters via Bayesian IRL.

        Overrides BaseEstimator.estimate() to use posterior standard
        deviations from MCMC instead of asymptotic standard errors.
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
        n_params = utility.num_parameters

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=result.log_likelihood,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * result.log_likelihood + 2 * n_params,
            bic=-2 * result.log_likelihood
            + n_params * torch.log(torch.tensor(n_obs)).item(),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, result.policy
            ),
        )

        # Use posterior std as standard errors
        posterior_std = result.metadata["posterior_std"]

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=utility.parameter_names,
            standard_errors=posterior_std,
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
