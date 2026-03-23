"""Base protocol and utilities for estimation algorithms.

This module defines the Estimator protocol that all estimation methods
must implement, ensuring a consistent interface across different
algorithms (NFXP, CCP, MaxEnt IRL, etc.).

The design allows for easy extensibility while maintaining a unified
API for users.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from econirl.core.types import DDCProblem, Panel
from econirl.inference.results import EstimationSummary
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import UtilityFunction


@dataclass
class EstimationResult:
    """Raw result from optimization (before computing inference).

    This is the internal result passed between optimization and inference.
    Users typically interact with EstimationSummary instead.

    Attributes:
        parameters: Optimized parameter values
        log_likelihood: Maximized log-likelihood
        value_function: Converged value function V(s)
        policy: Optimal choice probabilities π(a|s)
        hessian: Hessian at optimum (for standard errors)
        converged: Whether optimization converged
        num_iterations: Number of optimization iterations
        message: Convergence message from optimizer
    """

    parameters: torch.Tensor
    log_likelihood: float
    value_function: torch.Tensor
    policy: torch.Tensor
    hessian: torch.Tensor | None = None
    gradient_contributions: torch.Tensor | None = None
    converged: bool = True
    num_iterations: int = 0
    num_function_evals: int = 0
    num_inner_iterations: int = 0
    message: str = ""
    optimization_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Estimator(Protocol):
    """Protocol defining the interface for DDC estimators.

    All estimation algorithms (NFXP, CCP, MaxEnt, etc.) must implement
    this protocol to ensure consistent usage across the package.

    The main entry point is the `estimate()` method, which takes data
    and model specification and returns an EstimationSummary with
    rich statistical inference.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the estimation method."""
        ...

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate utility parameters from panel data.

        Args:
            panel: Panel data with observed choices
            utility: Utility function specification
            problem: DDCProblem with structural parameters
            transitions: Transition matrices P(s'|s,a)
            **kwargs: Estimator-specific options

        Returns:
            EstimationSummary with estimates and inference
        """
        ...


class BaseEstimator(ABC):
    """Abstract base class for DDC estimators.

    Provides common functionality and enforces the Estimator protocol.
    Subclasses must implement the core estimation logic.
    """

    def __init__(
        self,
        se_method: SEMethod = "asymptotic",
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the estimator.

        Args:
            se_method: Method for computing standard errors
            compute_hessian: Whether to compute Hessian for inference
            verbose: Whether to print progress messages
        """
        self._se_method = se_method
        self._compute_hessian = compute_hessian
        self._verbose = verbose

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the estimation method."""
        ...

    @abstractmethod
    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Core optimization routine. Must be implemented by subclasses.

        Args:
            panel: Panel data
            utility: Utility specification
            problem: Problem specification
            transitions: Transition matrices
            initial_params: Starting point for optimization
            **kwargs: Additional options

        Returns:
            EstimationResult with optimization output
        """
        ...

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate utility parameters from panel data.

        This method:
        1. Runs the optimization routine
        2. Computes standard errors
        3. Runs identification diagnostics
        4. Packages results into EstimationSummary

        Args:
            panel: Panel data with observed choices
            utility: Utility function specification
            problem: DDCProblem with structural parameters
            transitions: Transition matrices P(s'|s,a)
            initial_params: Starting values (optional)
            **kwargs: Estimator-specific options

        Returns:
            EstimationSummary with rich inference
        """
        import time

        from econirl.inference.identification import check_identification
        from econirl.inference.results import GoodnessOfFit
        from econirl.inference.standard_errors import compute_standard_errors

        start_time = time.time()

        # Run optimization
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        # Compute standard errors
        if result.hessian is not None or self._se_method == "bootstrap":
            # For bootstrap, create a function that re-estimates on a new panel
            estimate_fn = None
            if self._se_method == "bootstrap":
                def _bootstrap_estimate_fn(bootstrap_panel: Panel) -> torch.Tensor:
                    """Re-estimate on bootstrap sample (silent, fast settings)."""
                    bootstrap_result = self._optimize(
                        panel=bootstrap_panel,
                        utility=utility,
                        problem=problem,
                        transitions=transitions,
                        initial_params=result.parameters,  # Warm start
                    )
                    return bootstrap_result.parameters
                estimate_fn = _bootstrap_estimate_fn

            se_result = compute_standard_errors(
                parameters=result.parameters,
                hessian=result.hessian,
                gradient_contributions=result.gradient_contributions,
                panel=panel,
                method=self._se_method,
                estimate_fn=estimate_fn,
            )
            standard_errors = se_result.standard_errors
            variance_covariance = se_result.variance_covariance
        else:
            standard_errors = torch.full_like(result.parameters, float("nan"))
            variance_covariance = None

        # Identification diagnostics
        if result.hessian is not None:
            identification = check_identification(
                result.hessian, utility.parameter_names
            )
        else:
            identification = None

        # Goodness of fit
        n_obs = panel.num_observations
        n_params = utility.num_parameters
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

        total_time = time.time() - start_time

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=utility.parameter_names,
            standard_errors=standard_errors,
            hessian=result.hessian,
            variance_covariance=variance_covariance,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=ll,
            goodness_of_fit=goodness_of_fit,
            identification=identification,
            converged=result.converged,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=total_time,
            metadata=result.metadata,
        )

    def _compute_prediction_accuracy(
        self, panel: Panel, policy: torch.Tensor
    ) -> float:
        """Compute fraction of correctly predicted choices.

        Args:
            panel: Observed panel data
            policy: Estimated choice probabilities π(a|s)

        Returns:
            Fraction of observations where modal prediction matches choice
        """
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        predicted = policy[all_states].argmax(dim=1)
        total = all_states.shape[0]
        correct = (predicted == all_actions).sum().item()
        return correct / total if total > 0 else 0.0

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self._verbose:
            print(f"[{self.name}] {message}")
