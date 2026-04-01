"""Estimation results with rich statistical inference.

This module provides the EstimationSummary class, which presents estimation
results in a StatsModels-style format with standard errors, confidence
intervals, and hypothesis tests.

The goal is to provide economists with familiar, publication-ready output
that matches the conventions of the structural estimation literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy import stats


@dataclass
class IdentificationDiagnostics:
    """Diagnostics for parameter identification.

    Attributes:
        hessian_condition_number: Condition number of the Hessian matrix
        min_eigenvalue: Smallest eigenvalue of the Hessian
        max_eigenvalue: Largest eigenvalue of the Hessian
        rank: Numerical rank of the Hessian
        is_positive_definite: Whether Hessian is positive definite
        status: Human-readable identification status
    """

    hessian_condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    rank: int
    is_positive_definite: bool
    status: str


@dataclass
class GoodnessOfFit:
    """Goodness of fit measures.

    Attributes:
        log_likelihood: Maximized log-likelihood value
        num_parameters: Number of estimated parameters
        num_observations: Number of observations
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        pseudo_r_squared: McFadden's pseudo R-squared
        prediction_accuracy: Fraction of correctly predicted choices
    """

    log_likelihood: float
    num_parameters: int
    num_observations: int
    aic: float
    bic: float
    pseudo_r_squared: float | None = None
    prediction_accuracy: float | None = None


@dataclass
class EstimationSummary:
    """Rich estimation results with statistical inference.

    This class provides a StatsModels-style interface for presenting
    estimation results, including:
    - Point estimates with standard errors
    - Confidence intervals
    - Hypothesis tests (t-tests, Wald tests)
    - Identification diagnostics
    - Goodness of fit measures
    - Publication-ready output (summary tables, LaTeX)

    Attributes:
        parameters: Estimated parameter values
        parameter_names: Names of parameters
        standard_errors: Standard errors of estimates
        hessian: Hessian matrix at optimum (for inference)
        method: Estimation method used
        convergence_info: Details about optimization convergence

    Example:
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.summary())
        >>> result.to_latex("table.tex")
    """

    # Core estimates
    parameters: jnp.ndarray
    parameter_names: list[str]
    standard_errors: jnp.ndarray

    # Inference components
    hessian: jnp.ndarray | None = None
    variance_covariance: jnp.ndarray | None = None

    # Model info
    method: str = "Unknown"
    num_observations: int = 0
    num_individuals: int = 0
    num_periods: int = 0

    # Structural parameters
    discount_factor: float = 0.9999
    scale_parameter: float = 1.0

    # Fit and diagnostics
    log_likelihood: float | None = None
    goodness_of_fit: GoodnessOfFit | None = None
    identification: IdentificationDiagnostics | None = None

    # Convergence
    converged: bool = True
    num_iterations: int = 0
    convergence_message: str = ""

    # Solution
    value_function: jnp.ndarray | None = None
    policy: jnp.ndarray | None = None

    # Metadata
    estimation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and compute derived quantities."""
        if len(self.parameters) != len(self.parameter_names):
            raise ValueError(
                f"parameters ({len(self.parameters)}) and parameter_names "
                f"({len(self.parameter_names)}) must have same length"
            )

        # Compute variance-covariance from Hessian if not provided
        if self.variance_covariance is None and self.hessian is not None:
            try:
                self.variance_covariance = jnp.linalg.inv(-self.hessian)
            except Exception:
                pass  # Hessian not invertible

    @property
    def num_parameters(self) -> int:
        """Number of estimated parameters."""
        return len(self.parameters)

    @property
    def t_statistics(self) -> jnp.ndarray:
        """T-statistics for each parameter (H0: θ = 0)."""
        return self.parameters / self.standard_errors

    @property
    def p_values(self) -> jnp.ndarray:
        """Two-sided p-values for t-tests (H0: θ = 0)."""
        t_stats = np.asarray(self.t_statistics)
        # Large sample: use normal approximation
        p_vals = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        return jnp.array(p_vals, dtype=jnp.float32)

    def confidence_interval(
        self, alpha: float = 0.05
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute confidence intervals for parameters.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * self.standard_errors
        lower = self.parameters - margin
        upper = self.parameters + margin
        return lower, upper

    def get_parameter(self, name: str) -> dict[str, float]:
        """Get detailed results for a single parameter.

        Args:
            name: Parameter name

        Returns:
            Dictionary with estimate, se, t-stat, p-value, CI
        """
        idx = self.parameter_names.index(name)
        lower, upper = self.confidence_interval()

        return {
            "estimate": float(self.parameters[idx]),
            "std_error": float(self.standard_errors[idx]),
            "t_statistic": float(self.t_statistics[idx]),
            "p_value": float(self.p_values[idx]),
            "ci_lower": float(lower[idx]),
            "ci_upper": float(upper[idx]),
        }

    def wald_test(
        self,
        R: jnp.ndarray,
        r: jnp.ndarray | None = None,
    ) -> dict[str, float]:
        """Perform a Wald test for linear restrictions.

        Tests H0: R @ θ = r against H1: R @ θ ≠ r

        Args:
            R: Restriction matrix of shape (num_restrictions, num_parameters)
            r: Restriction values of shape (num_restrictions,). Default is zeros.

        Returns:
            Dictionary with test statistic, degrees of freedom, and p-value
        """
        if r is None:
            r = jnp.zeros(R.shape[0])

        if self.variance_covariance is None:
            raise ValueError("Variance-covariance matrix required for Wald test")

        # Wald statistic: (Rθ - r)' [R V R']^{-1} (Rθ - r)
        diff = R @ self.parameters - r
        middle = R @ self.variance_covariance @ R.T
        wald_stat = float(diff @ jnp.linalg.inv(middle) @ diff)

        df = R.shape[0]
        p_value = 1 - stats.chi2.cdf(wald_stat, df)

        return {
            "statistic": wald_stat,
            "df": df,
            "p_value": p_value,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns: estimate, std_error, t_stat, p_value, ci_lower, ci_upper
        """
        lower, upper = self.confidence_interval()

        return pd.DataFrame({
            "estimate": np.asarray(self.parameters),
            "std_error": np.asarray(self.standard_errors),
            "t_statistic": np.asarray(self.t_statistics),
            "p_value": np.asarray(self.p_values),
            "ci_lower": np.asarray(lower),
            "ci_upper": np.asarray(upper),
        }, index=self.parameter_names)

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a StatsModels-style summary table.

        Args:
            alpha: Significance level for confidence intervals

        Returns:
            Formatted string with estimation results
        """
        width = 80
        sep = "=" * width

        lines = [
            sep,
            "Dynamic Discrete Choice Estimation Results".center(width),
            sep,
        ]

        # Model info
        info_left = [
            f"Method:                    {self.method}",
            f"No. Observations:          {self.num_observations:,}",
            f"No. Individuals:           {self.num_individuals:,}",
        ]
        info_right = [
            f"Discount Factor (β):       {self.discount_factor}",
            f"Scale Parameter (σ):       {self.scale_parameter}",
            f"Date:                      {self.timestamp[:10]}",
        ]

        for left, right in zip(info_left, info_right):
            lines.append(f"{left:<40}{right}")

        if self.log_likelihood is not None:
            lines.append(f"Log-Likelihood:            {self.log_likelihood:,.2f}")

        lines.append("-" * width)

        # Parameter table header
        ci_pct = int((1 - alpha) * 100)
        header = f"{'':20} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>8} {'[' + str(alpha/2):>8} {str(1-alpha/2) + ']':>8}"
        lines.append(header)
        lines.append("-" * width)

        # Parameter rows
        lower, upper = self.confidence_interval(alpha)
        for i, name in enumerate(self.parameter_names):
            coef = float(self.parameters[i])
            se = float(self.standard_errors[i])
            t = float(self.t_statistics[i])
            p = float(self.p_values[i])
            lo = float(lower[i])
            hi = float(upper[i])

            # Format p-value
            if p < 0.001:
                p_str = "0.000"
            else:
                p_str = f"{p:.3f}"

            row = f"{name:20} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p_str:>8} {lo:>8.4f} {hi:>8.4f}"
            lines.append(row)

        lines.append("-" * width)

        # Identification diagnostics
        if self.identification is not None:
            lines.append("Identification Diagnostics:")
            lines.append(f"  Hessian Condition Number:    {self.identification.hessian_condition_number:.1f}")
            lines.append(f"  Min Eigenvalue:              {self.identification.min_eigenvalue:.4f}")
            lines.append(f"  Status:                      {self.identification.status}")
            lines.append("")

        # Goodness of fit
        if self.goodness_of_fit is not None:
            lines.append("Goodness of Fit:")
            lines.append(f"  AIC:                         {self.goodness_of_fit.aic:.1f}")
            lines.append(f"  BIC:                         {self.goodness_of_fit.bic:.1f}")
            if self.goodness_of_fit.pseudo_r_squared is not None:
                lines.append(f"  Pseudo R²:                   {self.goodness_of_fit.pseudo_r_squared:.3f}")
            if self.goodness_of_fit.prediction_accuracy is not None:
                lines.append(f"  Prediction Accuracy:         {self.goodness_of_fit.prediction_accuracy:.1%}")

        lines.append(sep)

        return "\n".join(lines)

    def to_latex(
        self,
        filename: str | None = None,
        caption: str = "Estimation Results",
        label: str = "tab:estimation",
    ) -> str:
        """Generate a LaTeX table of results.

        Args:
            filename: If provided, write to this file
            caption: Table caption
            label: Table label for referencing

        Returns:
            LaTeX table as string
        """
        df = self.to_dataframe()

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lcccccc}",
            r"\hline\hline",
            r"Parameter & Estimate & Std. Error & $t$-stat & $p$-value & \multicolumn{2}{c}{95\% CI} \\",
            r"\hline",
        ]

        for name in self.parameter_names:
            row = df.loc[name]
            lines.append(
                f"{name} & {row['estimate']:.4f} & {row['std_error']:.4f} & "
                f"{row['t_statistic']:.2f} & {row['p_value']:.3f} & "
                f"[{row['ci_lower']:.4f}, & {row['ci_upper']:.4f}] \\\\"
            )

        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        latex = "\n".join(lines)

        if filename is not None:
            with open(filename, "w") as f:
                f.write(latex)

        return latex

    def diagnostics(self) -> dict[str, Any]:
        """Bundle all available diagnostics into a single dict.

        Returns a dict with sections for goodness of fit, identification,
        numerical quality, and convergence. Each section contains the
        diagnostics that are available for this estimation result.
        Missing diagnostics are omitted rather than set to None.

        Returns:
            Dict with section keys: goodness_of_fit, identification,
            numerical_quality, convergence.
        """
        result: dict[str, Any] = {}

        # Goodness of fit
        if self.goodness_of_fit is not None:
            gof = self.goodness_of_fit
            result["goodness_of_fit"] = {
                "log_likelihood": gof.log_likelihood,
                "aic": gof.aic,
                "bic": gof.bic,
                "num_parameters": gof.num_parameters,
                "num_observations": gof.num_observations,
            }
            if gof.pseudo_r_squared is not None:
                result["goodness_of_fit"]["pseudo_r_squared"] = gof.pseudo_r_squared
            if gof.prediction_accuracy is not None:
                result["goodness_of_fit"]["prediction_accuracy"] = gof.prediction_accuracy

        # Identification
        if self.identification is not None:
            ident = self.identification
            result["identification"] = {
                "condition_number": ident.hessian_condition_number,
                "min_eigenvalue": ident.min_eigenvalue,
                "max_eigenvalue": ident.max_eigenvalue,
                "rank": ident.rank,
                "is_positive_definite": ident.is_positive_definite,
                "status": ident.status,
            }

        # Numerical quality
        num_quality: dict[str, Any] = {}
        if self.hessian is not None:
            eigenvalues = np.sort(np.real(np.linalg.eigvals(np.asarray(-self.hessian))))
            num_quality["hessian_eigenvalues"] = eigenvalues.tolist()
            num_quality["hessian_condition_number"] = (
                float(eigenvalues[-1] / eigenvalues[0])
                if eigenvalues[0] > 0 else float("inf")
            )
        num_quality["converged"] = self.converged
        num_quality["num_iterations"] = self.num_iterations
        num_quality["convergence_message"] = self.convergence_message
        num_quality["estimation_time"] = self.estimation_time
        if num_quality:
            result["numerical_quality"] = num_quality

        # Convergence
        result["convergence"] = {
            "converged": self.converged,
            "num_iterations": self.num_iterations,
            "message": self.convergence_message,
        }

        return result

    def __repr__(self) -> str:
        return (
            f"EstimationSummary(method='{self.method}', "
            f"n_params={self.num_parameters}, "
            f"converged={self.converged})"
        )

    def __str__(self) -> str:
        return self.summary()
