## Estimator: NNES (Neural Network Efficient Estimator)
## Paper(s): Nguyen 2025 "Neural Network Estimators for Dynamic Discrete Choice Models" (working paper). Doclinged at `papers/foundational/nguyen_2025_nnes.md`.
## Code: `src/econirl/estimation/nnes.py` -- exposes both `NNESEstimator` (NPL-based, default) and `NNESNFXPEstimator` (NFXP-based, legacy).

### Known-truth migration status

- Status: **validated**.
- RTD front door: `docs/estimators/nnes.md`.
- Dedicated TeX/PDF: `papers/econirl_package/primers/nnes/nnes.tex`.
- Result generator: `papers/econirl_package/primers/nnes/nnes_run.py`.
- Shared DGP harness: `experiments/known_truth.py`.
- Fast test: `tests/test_known_truth.py::test_nnes_smoke_fit_produces_known_truth_metrics_and_gates`.
- Component tests: `tests/test_nnes_known_truth_components.py`.

The current generator enforces gates. The previous diagnostic failure in this
audit was superseded by the NPL profiling, anchoring, and validation fixes now
present in the live artifacts.

### Loss / objective

- Paper target: an NPL-style estimator for dynamic discrete choice models that
  replaces grid-based policy evaluation with a neural value approximation. The
  architecture is designed to preserve the NPL zero-Jacobian property, making
  the likelihood score Neyman-orthogonal to first-stage value approximation
  error at the fixed point.

- Code path: `NNESEstimator` is the package default. For fixed CCPs, it
  profiles the policy-evaluation equation, maximizes the pseudo-likelihood over
  structural reward parameters, and then updates CCPs through the implied
  softmax policy-improvement map. The finite-state known-truth path computes
  exact profiled value components for the structural likelihood and trains the
  value network on the same target for diagnostics and large-state
  approximation support.

- Legacy path: `NNESNFXPEstimator` trains a neural value function against the
  NFXP soft Bellman operator. That path is explicitly documented as legacy and
  does not carry the paper's orthogonality claim.

- Match: **yes** for the default `NNESEstimator` path used in validation.

### Gradient / orthogonality

- Paper formula: the derivative of the policy-iteration map with respect to the
  value-function nuisance is zero at the true CCP fixed point. This gives a
  Neyman-orthogonal likelihood score and a block-diagonal information matrix
  for the structural parameters and first-stage value/CCP nuisance.

- Code implementation: the default estimator keeps the NPL outer loop, profiles
  value terms for each candidate policy, and updates CCPs with the model
  implied policy. The validation harness checks structural recovery rather than
  only value-network fit.

- Match: **yes** for the validated NPL-based estimator. The legacy NFXP-based
  class is not used for the success claim.

### Bellman / inner loop

- Paper algorithm: policy evaluation requires the transition law or a transition
  model, then approximates the value function with a neural network instead of
  solving a large grid DP at each outer step.

- Code algorithm: the known-truth validation supplies exact transition matrices
  from the DGP. This is paper-consistent because NNES is model-based in the
  current paper. Model-free NNES is explicitly a future extension, not the
  validated claim.

- Match: **yes**.

### Identification assumptions

- Paper conditions: the DDC model has Type-I extreme-value shocks, observed
  states/actions, a correctly specified transition model for policy evaluation,
  and value-function approximation accurate enough for the orthogonality
  argument. The package validation focuses on finite-dimensional structural
  reward parameters in known reward features.

- Code enforcement: the known-truth cells use homogeneous rewards, known
  transitions, action-dependent reward features with full rank, adequate
  state-action support, and exact solver truth for reward, policy, value, Q,
  and counterfactual comparisons.

- Match: **yes** for the current validation surface.

### Hyperparameter defaults vs paper defaults

Validation config is inherited from the known-truth harness and primer runner:

- default estimator: `NNESEstimator`;
- outer NPL iterations: 3;
- anchored value network with `anchor_state=0`;
- known transitions supplied by the synthetic DGP;
- enforced gates in `nnes_run.py`.

The public estimator constructor is unchanged.

### Current gated artifacts

Current primer artifact:
`papers/econirl_package/primers/nnes/nnes_results.json`.

Low-dimensional sanity cell:

- Cell: `canonical_low_action`.
- Result: **Pass, 11/11 gates**.
- Parameter cosine: 0.998240.
- Parameter relative RMSE: 0.065179.
- Reward RMSE: 0.010210.
- Policy TV: 0.005646.
- Value RMSE: 0.019845.
- Q RMSE: 0.023370.
- Type A/B/C regret: 0.000224, 0.000332, 0.000055.

High-dimensional primary cell:

- Cell: `canonical_high_action`.
- Result: **Pass, 11/11 gates**.
- Parameter cosine: 0.991204.
- Parameter relative RMSE: 0.135110.
- Reward RMSE: 0.064012.
- Policy TV: 0.023834.
- Value RMSE: 0.115620.
- Q RMSE: 0.137145.
- Type A/B/C regret: 0.004865, 0.005559, 0.001314.

### Interpretation

NNES now shows the intended estimator power on the current known-truth surface:
it recovers structural reward parameters and transports to counterfactuals in
both the low-dimensional sanity cell and the high-dimensional primary cell. The
claim is still bounded by the paper and harness assumptions: the current
validated package estimator is model-based and uses known or pre-estimated
transitions for policy evaluation.

### Findings / fixes applied

- The old failure mode in this audit is stale. It came from an earlier
  diagnostic run where NNES fit observed policy surfaces but failed structural
  reward and counterfactual recovery.
- The live artifacts now enforce gates and pass both canonical cells.
- RTD prose already reports the validation boundary and the high-dimensional
  primary-cell numbers.

- VALIDATION_LOG.md status: **Pass (low + high known-truth gates)**.
