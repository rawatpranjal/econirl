## Estimator: MCE-IRL Deep (neural reward variant of MCE-IRL)
## Paper(s): Ziebart 2010 plus Wulfmeier, Ondruska, and Posner 2015 "Maximum Entropy Deep Inverse Reinforcement Learning"
## Code: `src/econirl/estimators/mceirl_neural.py`; known-truth adapter in `experiments/known_truth.py`

### Status

- Status: **validated for anchored nonlinear reward-map recovery**.
- Result artifact: `papers/econirl_package/primers/deep_mce_irl/deep_mce_irl_results.json`.
- Primer: `papers/econirl_package/primers/deep_mce_irl/deep_mce_irl.pdf`.

### Loss / objective

- Paper formula: Wulfmeier et al. use a neural network from supplied state
  features to state rewards, then train it with the same MaxEnt/MCE occupancy
  gradient as the linear reward model.
- Implementation reference check: the `imitation.algorithms.mce_irl` reference
  implementation is finite-horizon tabular MCE and computes a state reward
  vector; its source contains a TODO for non-state-only rewards. This supports
  a narrow validation claim unless the econirl action-dependent extension is
  explicitly anchored.
- Code implementation: `MCEIRLNeural` trains a neural reward with surrogate
  loss `sum_{s,a} R_psi(s,a) [mu_pi(s,a)-mu_D(s,a)]`. Empirical and model
  moments are normalized discounted state-action occupancy measures using the
  empirical initial distribution.
- Match: **yes** for the Deep MCE occupancy-gradient objective.

### Identification and gated artifact

- Neural network weights are not finite economic parameters.
- For frozen neural reward truth, the gated artifact is the learned raw
  `reward_matrix`, with action 0 anchored at zero in Shapeshifter
  action-dependent cells.
- For finite linear reward truth, projected parameters are gated only when the
  supplied feature matrix is numerically identifiable. The neural-feature
  finite-theta cell is ill conditioned, so its poor projected-theta cosine/RMSE
  are reported as diagnostics rather than claimed as structural theta recovery.
- Counterfactual gates use the same affine IRL reward normalization convention
  before applying intervention deltas.
- Raw FCNN/Objectworld-style spatial feature learning is not implemented here.
  That would require a convolutional reward network over raw spatial inputs;
  the validated target is an MLP over supplied encodings.

### Gates

For frozen neural reward truth, the non-smoke gates require:

- convergence;
- occupancy residual <= 0.03;
- normalized reward RMSE <= 0.15;
- policy TV <= 0.05;
- normalized value RMSE <= 0.15;
- normalized Q RMSE <= 0.15;
- Type A/B/C counterfactual regret <= 0.08.

For finite linear truths, parameter cosine/RMSE gates are added only when the
projection condition number is at most 100. Otherwise reward, policy, value, Q,
occupancy, and counterfactual gates remain active.

### Current Results

- `canonical_low_state_only`: 11/11 gates pass.
- `deep_mce_neural_reward`: 9/9 gates pass. This is the primary validation
  cell.
- `deep_mce_neural_features`: 9/9 active gates pass. Projected-theta cosine is
  diagnostic because the projection condition number is about 479.
- `deep_mce_neural_reward_features`: 9/9 gates pass.

### Findings / fixes applied

- Added a Shapeshifter known-truth bridge with no absorbing-state assumption.
- Added full reward/policy/value/Q/occupancy/counterfactual masks for DGPs
  without an absorbing state or exit action.
- Added action-0 reward anchoring for Shapeshifter action-dependent neural
  reward cells.
- Added a Shapeshifter reward-scale knob for non-degenerate neural reward
  signal while keeping default behavior unchanged.
- Routed neural-reward validation through `summary.metadata["reward_matrix"]`
  and empty finite theta.
- Kept projected theta diagnostic unless the finite feature basis is
  numerically identifiable.
