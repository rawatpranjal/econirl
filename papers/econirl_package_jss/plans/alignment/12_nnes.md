## Estimator: NNES (Neural Network Efficient Estimator)
## Paper(s): Nguyen 2025 "Neural Network Estimators for Dynamic Discrete Choice Models" (working paper). Doclinged at `papers/foundational/nguyen_2025_nnes.md`.
## Code: `src/econirl/estimation/nnes.py` — exposes both `NNESEstimator` (NPL-based, default) and `NNESNFXPEstimator` (NFXP-based, legacy).

### Known-truth migration status

- Status: **diagnostic, not validated**.
- RTD front door: `docs/estimators/nnes.md`.
- Dedicated TeX/PDF: `papers/econirl_package/primers/nnes/nnes.tex`.
- Result generator: `papers/econirl_package/primers/nnes/nnes_run.py`.
- Shared DGP harness: `experiments/known_truth.py`.
- Fast test: `tests/test_known_truth.py::test_nnes_smoke_fit_produces_known_truth_metrics_and_gates`.

The implementation now runs through the shared known-truth harness and emits a
full primer artifact, but the canonical non-smoke run fails hard structural
recovery. The generator uses `enforce_gates=False` so the failure diagnostics
are written instead of being hidden by an exception.

### Loss / objective

- Paper target: an NPL-style estimator with a neural value approximation and a
  zero-Jacobian/Neyman-orthogonality property at the true CCP fixed point.
- Code path: `NNESEstimator` trains a ReLU value network on the Hotz-Miller NPL
  value target, then maximizes a pseudo-likelihood over structural reward
  parameters and updates CCPs from the implied policy.
- Legacy path: `NNESNFXPEstimator` trains on a soft Bellman residual and does
  not have the same NPL orthogonality claim.

The code structure matches the intended NPL neural-value architecture, but the
known-truth recovery result shows that this is not enough for validation.

### Fixes applied

- The NPL value target is now normalized by the configured anchor state before
  value-network training. This aligns the target with the anchored network
  output.
- The reported validation value function is re-evaluated from the final
  recovered reward and policy on the Bellman scale. The anchored neural values
  remain in `summary.metadata["v_network_values"]`.
- NNES has a harness contract, compatibility checks, smoke test, non-smoke
  recovery gates, and regenerated tutorial artifacts.

### Canonical diagnostic result

Cell: `canonical_low_action`, 2,000 individuals, 80 periods, 21 states, 3
actions, action-dependent low-dimensional reward features.

Current gates:

- NPL outer iterations: pass.
- final V-network loss: pass.
- baseline policy TV: pass.
- parameter cosine: fail.
- parameter relative RMSE: fail.
- reward RMSE: fail.
- value RMSE: fail.
- Q RMSE: fail.
- Type A/B/C counterfactual regret: fail.

Current metrics:

- final V-network loss: 0.001059.
- parameter cosine: 0.585979.
- parameter relative RMSE: 3.422698.
- reward RMSE: 0.514228.
- baseline policy TV: 0.014661.
- value RMSE: 2.814636.
- Q RMSE: 2.793261.
- Type A/B/C regret: 7.995630, 3.930581, 10.870553.

### Interpretation

NNES can fit the observed-policy surface on the canonical DGP while recovering
the wrong reward parameters and failing counterfactual transport. That is not a
validated structural estimator result. Future work should debug the NPL update
formulation, outer-loop stability, and the interaction between finite-sample
CCPs, value-network approximation, and structural MLE.
