# estimation/ - All Estimator Implementations

## Base Contract (`base.py`)

**BaseEstimator** subclasses implement `_optimize(panel, utility, problem, transitions, initial_params) -> EstimationResult`. The base `estimate()` method wraps this with SE computation, identification checks, and goodness-of-fit, returning an `EstimationSummary`.

**EstimationResult** fields: `parameters`, `log_likelihood`, `value_function`, `policy`, `hessian`, `gradient_contributions`, `converged`, `num_iterations`, `metadata`.

**Estimator** protocol requires: `name` property, `estimate(panel, utility, problem, transitions) -> EstimationSummary`.

## Forward Estimators

- **NFXPEstimator** (`nfxp.py`): Nested fixed point. L-BFGS outer loop, VI/hybrid inner loop. Classic Rust (1987) approach.
- **CCPEstimator** (`ccp.py`): Hotz-Miller CCP with Nested Pseudo-Likelihood (NPL) iterations.

## IRL Estimators

- **MaxEntIRLEstimator** (`maxent_irl.py`): Maximum entropy IRL (Ziebart 2008). Gradient descent on feature matching.
- **MCEIRLEstimator** (`mce_irl.py`): Maximum causal entropy IRL. Uses `MCEIRLConfig` for hyperparameters.
- **MaxMarginPlanningEstimator** (`max_margin_planning.py`): Max margin planning with `MMPConfig`.

## Neural / Deep Estimators

- **TDCCPEstimator** (`td_ccp.py`): CCP decomposition + neural approximate value iteration. Config: `TDCCPConfig`.
- **GLADIUSEstimator** (`gladius.py`): Q-network + EV-network with Bellman consistency penalty. Config: `GLADIUSConfig`.

## Adversarial Estimators (`adversarial/`)

- **GAILEstimator** (`adversarial/gail.py`): Generative adversarial imitation learning. Config: `GAILConfig`.
- **AIRLEstimator** (`adversarial/airl.py`): Adversarial IRL with reward recovery. Config: `AIRLConfig`.
- Shared: `TabularDiscriminator`, `LinearDiscriminator` in `adversarial/discriminator.py`.

## Generative Estimator

- **GCLEstimator** (`gcl.py`): Guided cost learning. Config: `GCLConfig`.

## Utilities

- `estimate_transition_probs(panel, problem)`: Estimate transition matrices from data.
- `estimate_transition_probs_by_group(panel, problem, groups)`: Group-wise estimation.

## Gotchas

- MCE IRL `_compute_expected_features()` MUST iterate over empirical states, not stationary distribution. See root CLAUDE.md.
- All estimators expect transitions shape `(num_actions, num_states, num_states)`.
- IRL rewards are identified only up to constants. Normalize parameters to unit norm for stable optimization.
- Config dataclasses (MCEIRLConfig, etc.) control hyperparameters like learning rate, inner iterations, and convergence tolerances.
