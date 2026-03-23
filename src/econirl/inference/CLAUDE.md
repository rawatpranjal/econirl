# inference/ - Standard Errors, Identification, Results

## Standard Errors (`standard_errors.py`)

`SEMethod = Literal["asymptotic", "robust", "bootstrap", "clustered"]`

**compute_standard_errors(parameters, hessian, gradient_contributions, panel, method, ...)** -> StandardErrorResult (contains `standard_errors`, `variance_covariance`, `method`, `details`).

Methods:
- **asymptotic**: `Var = (-H)^{-1}`. Uses progressive ridge regularization if Hessian is singular. Falls back to NaN.
- **robust** (sandwich): `Var = H^{-1} B H^{-1}` where `B = sum(g_i g_i')`. Requires `gradient_contributions`.
- **bootstrap**: Resamples individuals with replacement, re-estimates via `estimate_fn`. Requires `panel` and `estimate_fn`.
- **clustered**: Sums gradients within individuals before outer product. Small-sample correction: `G/(G-1) * (N-1)/(N-K)`.

**compute_numerical_hessian(params, ll_fn, eps)**: Central differences, four-point formula for mixed partials.

**compute_gradient_contributions(params, panel, log_prob_fn, eps)**: Per-observation numerical gradients.

## Identification (`identification.py`)

**check_identification(hessian, parameter_names, tol)** -> IdentificationDiagnostics. Eigenvalue analysis of `-H`. Reports: condition number, rank, positive definiteness, status string.

**diagnose_identification_issues(hessian, parameter_names)**: Returns list of diagnostic messages identifying problematic parameters, near-zero eigenvalues, high correlations (>0.95).

**check_local_identification(ll_fn, params, ...)**: Explores likelihood surface in random directions to confirm local maximum.

## Results (`results.py`)

**EstimationSummary**: Rich result object with StatsModels-style output.
- Properties: `t_statistics`, `p_values`, `confidence_interval(alpha)`
- Methods: `summary()` (formatted table), `to_dataframe()`, `to_latex()`, `wald_test(R, r)`, `get_parameter(name)`
- Contains: `goodness_of_fit` (GoodnessOfFit), `identification` (IdentificationDiagnostics), `policy`, `value_function`

**GoodnessOfFit**: `log_likelihood`, `aic`, `bic`, `prediction_accuracy`, `pseudo_r_squared`.

## Gotchas

- Asymptotic SEs can be NaN if Hessian is singular -- always check `identification.status`.
- Bootstrap requires `estimate_fn` callback, not just the Hessian.
- Clustered SEs require `gradient_contributions` of shape `(n_observations, n_params)` ordered consistently with panel trajectories.
