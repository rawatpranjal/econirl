# Deep MCE-IRL

**Reference PDF:** `papers/econirl_package/primers/deep_mce_irl/deep_mce_irl.pdf`.

Deep MCE-IRL uses the maximum causal entropy occupancy-matching objective with
a neural reward map. The validated package target is nonlinear reward-map
recovery over supplied state encodings and known transitions.

## Validation Status

**Pass.** The generated artifact comes from
`papers/econirl_package/primers/deep_mce_irl/deep_mce_irl_run.py` and the
shared known-truth harness.

The primary cell is `deep_mce_neural_reward`: Shapeshifter with a frozen
nonlinear neural reward, linear supplied features, stochastic known transitions,
32 states, 3 actions, 2,000 individuals, and 80 periods. Action 0 is anchored
at zero reward. It passes 9/9 gates on occupancy residual, normalized
reward/value/Q RMSE, policy TV, and Type A/B/C counterfactual regret.

The finite-theta projection is not the primary success artifact for neural
reward truth. Parameter gates are used only when the true reward is linear in a
well-conditioned supplied feature matrix. In the neural-feature linear-reward
cell, the projection is ill conditioned, so the poor projected-theta cosine is
reported as diagnostic rather than claimed as finite-theta recovery.

## Usage Scope

Use this path when transitions are known and the reward can be represented by an
MLP over supplied state encodings, with an explicit reward gauge such as an
anchor action or absorbing-state anchor. Do not treat raw neural weights as
identified structural parameters.

This is not raw FCNN/Objectworld-style spatial reward learning. That would
require a convolutional reward network over raw spatial inputs; the current
validated implementation learns from supplied encodings.

## Artifacts

- PDF source: `papers/econirl_package/primers/deep_mce_irl/deep_mce_irl.tex`
- Result generator: `papers/econirl_package/primers/deep_mce_irl/deep_mce_irl_run.py`
- Shared DGP harness: `experiments/known_truth.py`
- Results: `papers/econirl_package/primers/deep_mce_irl/deep_mce_irl_results.json`
