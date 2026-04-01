"""Side-by-side estimation comparison tables.

Provides etable() for generating stargazer-style comparison tables
from multiple EstimationSummary objects, with text, LaTeX, and HTML output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from econirl.inference.results import EstimationSummary


def etable(
    *models: EstimationSummary,
    model_names: list[str] | None = None,
    stars: bool = True,
    confidence_intervals: bool = False,
    stats: list[str] | None = None,
    float_format: str = "{:.4f}",
    output: str = "text",
) -> str:
    """Generate a side-by-side comparison table for multiple models.

    Produces a stargazer-style table comparing parameter estimates,
    standard errors (or confidence intervals), and fit statistics.

    Args:
        *models: Two or more EstimationSummary objects to compare.
        model_names: Display names for each model. Defaults to method names.
        stars: Add significance stars (*** p<0.01, ** p<0.05, * p<0.1).
        confidence_intervals: Show 95 percent CIs instead of standard errors.
        stats: Summary statistics to show at bottom. Default is
            n_obs, log_likelihood, aic, bic, pseudo_r2.
        float_format: Format string for numbers.
        output: Output format: 'text', 'latex', or 'html'.

    Returns:
        Formatted comparison table as string.
    """
    if len(models) < 1:
        raise ValueError("At least one model required")

    if model_names is None:
        model_names = [m.method for m in models]

    if stats is None:
        stats = ["n_obs", "log_likelihood", "aic", "bic", "pseudo_r2"]

    # Collect all parameter names in order of first appearance
    all_params = []
    seen = set()
    for m in models:
        for name in m.parameter_names:
            if name not in seen:
                all_params.append(name)
                seen.add(name)

    if output == "latex":
        return _etable_latex(
            models, model_names, all_params, stars,
            confidence_intervals, stats, float_format,
        )
    elif output == "html":
        return _etable_html(
            models, model_names, all_params, stars,
            confidence_intervals, stats, float_format,
        )
    else:
        return _etable_text(
            models, model_names, all_params, stars,
            confidence_intervals, stats, float_format,
        )


def _star_str(p_value: float) -> str:
    """Return significance stars for a p-value."""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    return ""


def _get_stat_value(model: EstimationSummary, stat: str) -> str:
    """Extract a summary statistic from a model."""
    if stat == "n_obs":
        return f"{model.num_observations:,}"
    elif stat == "log_likelihood" and model.log_likelihood is not None:
        return f"{model.log_likelihood:,.2f}"
    elif stat == "aic" and model.goodness_of_fit is not None:
        return f"{model.goodness_of_fit.aic:,.1f}"
    elif stat == "bic" and model.goodness_of_fit is not None:
        return f"{model.goodness_of_fit.bic:,.1f}"
    elif stat == "pseudo_r2" and model.goodness_of_fit is not None:
        if model.goodness_of_fit.pseudo_r_squared is not None:
            return f"{model.goodness_of_fit.pseudo_r_squared:.4f}"
    return ""


_STAT_LABELS = {
    "n_obs": "Observations",
    "log_likelihood": "Log-Likelihood",
    "aic": "AIC",
    "bic": "BIC",
    "pseudo_r2": "Pseudo R-squared",
}


def _etable_text(
    models, model_names, all_params, stars,
    confidence_intervals, stats, float_format,
) -> str:
    """Generate ASCII text comparison table."""
    col_width = 14
    label_width = 20
    n_models = len(models)

    sep = "=" * (label_width + n_models * col_width)
    dash = "-" * (label_width + n_models * col_width)

    lines = [sep]

    # Header row
    header = f"{'':>{label_width}}"
    for name in model_names:
        header += f"{name:>{col_width}}"
    lines.append(header)
    lines.append(sep)

    # Parameter rows
    for param in all_params:
        # Estimate row
        row = f"{param:>{label_width}}"
        for m in models:
            if param in m.parameter_names:
                idx = m.parameter_names.index(param)
                coef = float(m.parameters[idx])
                cell = float_format.format(coef)
                if stars:
                    p = float(m.p_values[idx])
                    cell += _star_str(p)
                row += f"{cell:>{col_width}}"
            else:
                row += f"{'':>{col_width}}"
        lines.append(row)

        # SE or CI row
        row = f"{'':>{label_width}}"
        for m in models:
            if param in m.parameter_names:
                idx = m.parameter_names.index(param)
                if confidence_intervals:
                    lower, upper = m.confidence_interval()
                    lo = float(lower[idx])
                    hi = float(upper[idx])
                    cell = f"[{lo:.3f}, {hi:.3f}]"
                else:
                    se = float(m.standard_errors[idx])
                    cell = f"({float_format.format(se)})"
                row += f"{cell:>{col_width}}"
            else:
                row += f"{'':>{col_width}}"
        lines.append(row)

    lines.append(dash)

    # Summary statistics
    for stat in stats:
        label = _STAT_LABELS.get(stat, stat)
        row = f"{label:>{label_width}}"
        for m in models:
            val = _get_stat_value(m, stat)
            row += f"{val:>{col_width}}"
        lines.append(row)

    lines.append(sep)

    if stars:
        lines.append("Note: * p<0.10, ** p<0.05, *** p<0.01")

    return "\n".join(lines)


def _etable_latex(
    models, model_names, all_params, stars,
    confidence_intervals, stats, float_format,
) -> str:
    """Generate LaTeX stargazer-style comparison table."""
    n_models = len(models)
    col_spec = "l" + "c" * n_models

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline\hline",
    ]

    # Header
    header = " & " + " & ".join(f"({i+1})" for i in range(n_models)) + r" \\"
    lines.append(header)
    header2 = " & " + " & ".join(model_names) + r" \\"
    lines.append(header2)
    lines.append(r"\hline")

    # Parameters
    for param in all_params:
        cells = []
        se_cells = []
        for m in models:
            if param in m.parameter_names:
                idx = m.parameter_names.index(param)
                coef = float(m.parameters[idx])
                cell = float_format.format(coef)
                if stars:
                    p = float(m.p_values[idx])
                    star = _star_str(p)
                    cell = f"${cell}^{{{star}}}$" if star else f"${cell}$"
                cells.append(cell)

                if confidence_intervals:
                    lower, upper = m.confidence_interval()
                    se_cells.append(f"$[{float(lower[idx]):.3f}, {float(upper[idx]):.3f}]$")
                else:
                    se = float(m.standard_errors[idx])
                    se_cells.append(f"$({float_format.format(se)})$")
            else:
                cells.append("")
                se_cells.append("")

        lines.append(f"{param} & " + " & ".join(cells) + r" \\")
        lines.append(" & " + " & ".join(se_cells) + r" \\")

    lines.append(r"\hline")

    # Stats
    for stat in stats:
        label = _STAT_LABELS.get(stat, stat)
        cells = [_get_stat_value(m, stat) for m in models]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    if stars:
        lines.append(r"\begin{tablenotes}")
        lines.append(r"\small \item Note: $^{*}$p$<$0.10; $^{**}$p$<$0.05; $^{***}$p$<$0.01")
        lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _etable_html(
    models, model_names, all_params, stars,
    confidence_intervals, stats, float_format,
) -> str:
    """Generate HTML comparison table."""
    lines = ['<table class="etable">']
    lines.append("<thead><tr>")
    lines.append("<th></th>")
    for name in model_names:
        lines.append(f"<th>{name}</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    for param in all_params:
        # Estimate row
        lines.append("<tr>")
        lines.append(f"<td><b>{param}</b></td>")
        for m in models:
            if param in m.parameter_names:
                idx = m.parameter_names.index(param)
                coef = float(m.parameters[idx])
                cell = float_format.format(coef)
                if stars:
                    p = float(m.p_values[idx])
                    cell += f"<sup>{_star_str(p)}</sup>"
                lines.append(f"<td>{cell}</td>")
            else:
                lines.append("<td></td>")
        lines.append("</tr>")

        # SE row
        lines.append('<tr class="se-row">')
        lines.append("<td></td>")
        for m in models:
            if param in m.parameter_names:
                idx = m.parameter_names.index(param)
                if confidence_intervals:
                    lower, upper = m.confidence_interval()
                    cell = f"[{float(lower[idx]):.3f}, {float(upper[idx]):.3f}]"
                else:
                    se = float(m.standard_errors[idx])
                    cell = f"({float_format.format(se)})"
                lines.append(f"<td>{cell}</td>")
            else:
                lines.append("<td></td>")
        lines.append("</tr>")

    # Stats
    lines.append('<tr class="separator"><td colspan="100"><hr></td></tr>')
    for stat in stats:
        label = _STAT_LABELS.get(stat, stat)
        lines.append("<tr>")
        lines.append(f"<td>{label}</td>")
        for m in models:
            val = _get_stat_value(m, stat)
            lines.append(f"<td>{val}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")

    if stars:
        lines.append('<p class="etable-note">Note: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01</p>')

    return "\n".join(lines)
