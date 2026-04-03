# Identification and Counterfactual Robustness (Simulation Study)

This page walks through the identification study used in the appendix and compares five IRL/DDC approaches plus NFXP on a controlled synthetic environment. The focus is reward recovery and counterfactual accuracy under transition changes (Type B).

## Problem setup

- Tabular MDP with 180 regular states (episode × content × wait) + 1 absorbing state.
- Three actions: `buy`, `wait`, `exit`. Discount factor `β=0.95`.
- True reward: `r*(s,buy)=u(c)-2.0`, `r*(s,wait)=u(c)-0.08·hours`, `r*(s,exit)=0`.
- Deterministic dynamics (content sequence known). See `experiments/identification/run_full_simulation.py`.

## Methods compared

- Reduced‑form logit over a flexible `Q(s,a)` (recovers advantages A*, not r).
- AIRL (Fu et al. 2018) without anchors (recovers shaped rewards r + γΦ(s′) − Φ(s)).
- AIRL with anchors (LSW): exit anchored to 0, V(absorbing)=0 ⇒ recovers r*.
- IQ‑Learn (Garg et al. 2021): inverse‑Bellman operator separates r from continuation values.
- GLADIUS (Kang et al. 2025): neural Q/EV with inverse‑Bellman extraction.
- NFXP (Rust 1987): structural MLE with nested fixed point; uses known transitions.

## What to expect

- Type A (state extrapolation): all methods agree (CCPs differ only by a state‑specific constant that cancels in softmax).
- Type B (transition change): Only structural‑reward methods transfer (AIRL+anchors, IQ/GLADIUS, NFXP). Reduced‑form and AIRL‑no‑anchors fail due to bundled continuation values or shaping.
- Shaping magnitude α → larger counterfactual error monotonically.

## Reproduce

- Population results + plots (drives appendix tables A–C, E–G):

```
make population
```

- Quick NFXP finite‑sample fill for Table D (append NFXP column to results):

```
make nfxp
```

- Full finite‑sample re‑run for all estimators (long):

```
make finite-sample
```

Artifacts live in `experiments/identification/results/`:

- `full_simulation.json` — single source of truth for population numbers and quick NFXP D results.
- `type_ii_by_k.png` — Type B CCP error vs skip magnitude k
- `shaping_sweep.png` — Type II error vs shaping α
- `anchor_misspec.png` — Type II error vs exit anchor misspec ε

## Results (population)

- Reward recovery (A): AIRL+anchors, IQ/GLADIUS, NFXP match oracle (MSE 0, Corr 1). AIRL‑no‑anchors and Reduced‑form do not.
- Type B (B): At k ∈ {2,3,5,7,10}, structural methods are exact (0.000); Reduced‑form ≈ 0.08–0.14, AIRL‑no‑anchors ≈ 0.014–0.025.
- Shaping sweep (C): α 0→1 increases error 0.000→0.027.
- Paywall (F): Reduced‑form 0.185, AIRL‑no‑anchors 0.009; structural methods 0.000.
- State‑only (G): AIRL state‑only transfers perfectly (Type II 0.000), as predicted by Fu et al. (Theorem 5.1).

## Results (finite sample, NFXP quick pass)

From `make nfxp` (3 seeds per N, Type B k=3):

- NFXP Type II CCP error: N=200 → 0.0305, 500 → 0.0312, 2k → 0.0313, 5k → 0.0311, 10k → 0.0309.

Note: The full D table includes all estimators; run `make finite-sample` to populate every column.

## New metrics

The study also computes complementary metrics per counterfactual:

- CCP divergences (uniform and occupancy‑weighted): L1, Linf, TV, KL.
- Policy regret (uniform and occupancy‑weighted expected return gap).

See `experiments/identification/metrics.py` and `results.full_simulation.json → metrics`.

## Why this matters

- Counterfactual prediction with dynamics changes requires disentangled rewards. The anchors (LSW/AIRL) or inverse‑Bellman operators (IQ/GLADIUS) or structural MLE (NFXP) supply the “additional restrictions” necessary (Kalouptsidi, Scott, Souza 2021). Reduced‑form advantage models cannot separate flow utility from continuation values and fail on Type B.

## References

- Rust (1987) Econometrica — NFXP; Iskhakov et al. (2016) SA→NK polyalgorithm
- Fu et al. (2018) ICLR — AIRL and disentanglement (state‑only)
- Garg et al. (2021) NeurIPS — IQ‑Learn inverse soft‑Q
- Kalouptsidi, Scott, Souza (2021) Quantitative Economics — counterfactual identification

