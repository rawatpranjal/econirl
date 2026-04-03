"""Base mixin for neural reward estimators (NeuralGLADIUS, NeuralAIRL).

Neural reward estimators learn R(s,a,ctx) via neural networks, then project
onto linear features to extract approximate structural parameters theta.
Key difference from structural estimators: the NN approximates the REWARD
(or Q-function), not the value function directly.

The projection step uses least-squares regression of implied neural rewards
onto the user-provided feature matrix, producing interpretable theta with
pseudo standard errors and an R-squared quality metric.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as scipy_norm


class NeuralEstimatorMixin:
    """Shared functionality for neural reward estimators.

    Provides:
    - Least-squares projection of neural rewards onto linear features
    - Pseudo p-value computation from projected theta and SEs
    - Formatted summary output with projection R-squared warning
    """

    def _to_numpy(self, values: object) -> np.ndarray:
        """Convert torch, JAX, or numpy-like values to a NumPy array."""
        if hasattr(values, "detach") and hasattr(values, "cpu"):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    def _project_parameters(
        self,
        features: object,
        rewards: object,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Project neural rewards onto linear features via least-squares.

        Solves theta = argmin ||Phi @ theta - r||^2 where Phi is the feature
        matrix and r is the vector of implied rewards from the neural network.

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix of shape (N, K) for observed transitions.
        rewards : torch.Tensor
            Implied rewards of shape (N,) from the neural network.

        Returns
        -------
        theta : torch.Tensor
            Projected parameters of shape (K,).
        se : torch.Tensor
            Pseudo standard errors of shape (K,).
        r_squared : float
            R-squared of the projection, indicating how well the linear
            feature model approximates the neural reward surface.
        """
        features_np = self._to_numpy(features).astype(np.float32, copy=False)
        rewards_np = self._to_numpy(rewards).astype(np.float32, copy=False).reshape(-1)

        N, K = features_np.shape

        theta, _, _, _ = np.linalg.lstsq(features_np, rewards_np, rcond=None)

        predicted = features_np @ theta
        residuals = rewards_np - predicted
        ss_res = float((residuals**2).sum())
        ss_tot = float(((rewards_np - rewards_np.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        sigma2 = ss_res / max(N - K, 1)
        try:
            cov = sigma2 * np.linalg.inv(features_np.T @ features_np)
            se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
        except Exception:
            se = np.full(K, float("nan"), dtype=np.float32)

        return theta, se, r_squared

    def _compute_pvalues(
        self,
        params_dict: dict[str, float],
        se_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute p-values from projected theta and pseudo standard errors.

        Uses a two-sided Wald test: t = theta / se, p = 2 * (1 - Phi(|t|)).

        Parameters
        ----------
        params_dict : dict
            Parameter point estimates {name: value}.
        se_dict : dict
            Standard errors {name: value}.

        Returns
        -------
        dict
            P-values {name: pvalue}.
        """
        pvalues: dict[str, float] = {}
        for name in params_dict:
            se = se_dict.get(name, float("nan"))
            if np.isfinite(se) and se > 0:
                t = params_dict[name] / se
                pvalues[name] = float(2 * (1 - scipy_norm.cdf(abs(t))))
            else:
                pvalues[name] = float("nan")
        return pvalues

    def _format_neural_summary(
        self,
        method_name: str,
        params: dict[str, float] | None = None,
        se: dict[str, float] | None = None,
        pvalues: dict[str, float] | None = None,
        projection_r2: float | None = None,
        n_observations: int | None = None,
        n_epochs: int | None = None,
        converged: bool | None = None,
        discount: float | None = None,
        scale: float | None = None,
        context_dim: int | None = None,
        extra_lines: list[str] | None = None,
    ) -> str:
        """Format summary for neural estimators with projection R-squared.

        Parameters
        ----------
        method_name : str
            Name of the estimation method (e.g., "NeuralGLADIUS").
        params : dict, optional
            Parameter estimates.
        se : dict, optional
            Standard errors.
        pvalues : dict, optional
            P-values.
        projection_r2 : float, optional
            R-squared of the feature projection.
        n_observations : int, optional
            Number of observations used.
        n_epochs : int, optional
            Number of training epochs.
        converged : bool, optional
            Whether training converged.
        discount : float, optional
            Discount factor.
        scale : float, optional
            Scale parameter.
        context_dim : int, optional
            Context dimension.
        extra_lines : list of str, optional
            Additional info lines to include.

        Returns
        -------
        str
            Formatted summary string.
        """
        width = 70
        lines: list[str] = []
        lines.append("=" * width)
        lines.append(f"  {method_name} Estimation Results")
        lines.append("=" * width)

        # Model info
        if n_observations is not None:
            lines.append(f"  Observations:    {n_observations}")
        if discount is not None:
            lines.append(f"  Discount:        {discount}")
        if scale is not None:
            lines.append(f"  Scale:           {scale}")
        if context_dim is not None and context_dim > 0:
            lines.append(f"  Context dim:     {context_dim}")
        if n_epochs is not None:
            lines.append(f"  Training epochs: {n_epochs}")
        if converged is not None:
            lines.append(f"  Converged:       {converged}")
        lines.append("-" * width)

        # Projection R-squared
        if projection_r2 is not None:
            r2_str = f"{projection_r2:.4f}"
            lines.append(f"  Projection R2:   {r2_str}")
            if projection_r2 < 0.95:
                lines.append(
                    "  WARNING: R2 < 0.95 -- linear features may not fully"
                )
                lines.append(
                    "  capture the neural reward surface. Interpret theta"
                )
                lines.append("  with caution.")
            lines.append("-" * width)

        # Parameter table
        if params is not None:
            header = f"  {'Parameter':<16} {'Estimate':>12} {'Std.Err':>12} {'P-value':>12}"
            lines.append(header)
            lines.append("  " + "-" * (width - 4))
            for name in params:
                est = params[name]
                se_val = se.get(name, float("nan")) if se else float("nan")
                pv = pvalues.get(name, float("nan")) if pvalues else float("nan")

                est_str = f"{est:12.6f}"
                se_str = f"{se_val:12.6f}" if np.isfinite(se_val) else f"{'NaN':>12}"
                pv_str = f"{pv:12.4f}" if np.isfinite(pv) else f"{'NaN':>12}"

                lines.append(f"  {name:<16} {est_str} {se_str} {pv_str}")
            lines.append("-" * width)
        else:
            lines.append("  No feature projection (params_ is None)")
            lines.append(
                "  Pass features= to fit() to extract structural parameters."
            )
            lines.append("-" * width)

        # Extra info
        if extra_lines:
            for line in extra_lines:
                lines.append(f"  {line}")
            lines.append("-" * width)

        lines.append(
            "  Note: SEs are pseudo standard errors from the projection"
        )
        lines.append(
            "  regression, NOT from the structural likelihood."
        )
        lines.append("=" * width)

        return "\n".join(lines)
