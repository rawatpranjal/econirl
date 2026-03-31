# Guide

This guide explains why each of econirl's 9 core estimators exists, what theorem makes it unique, and when to use it. Every estimator occupies a point in the capability space that no other method covers.

## Quick Reference

| Estimator | Direction | Reward Type | Needs Transitions | Standard Errors | Scales Beyond Tabular | Transfer |
|-----------|-----------|-------------|-------------------|-----------------|----------------------|----------|
| NFXP-NK | Forward (θ→π) | Linear | Yes | Analytical (MLE) | No | No |
| CCP | Forward (θ→π) | Linear | Yes | Hessian | No | No |
| MCE-IRL | Inverse (π→θ) | Linear / Neural | Yes | Bootstrap | Deep variant only | No |
| TD-CCP | Forward (θ→π) | Linear | Yes | Hessian | Yes (neural AVI) | No |
| NNES | Forward (θ→π) | Linear | Yes | Valid (orthogonality) | Yes (neural V) | No |
| SEES | Forward (θ→π) | Linear | Yes | Marginal Hessian | Yes (O(1) in \|S\|) | No |
| AIRL | Inverse (π→R) | Linear / Tabular | Yes | No | No | Yes |
| f-IRL | Inverse (π→R) | Tabular | Yes | No | No | No |
| BC | Imitation | None | No | No | No | No |

## Decision Flowchart

```
Do you have a parametric utility model u(s,a;θ)?
├── YES (structural estimation — recover θ)
│   ├── Tabular state space (|S| < 10K)?
│   │   ├── Need MLE efficiency + analytical SEs? → NFXP-NK
│   │   └── Need speed / model selection / games? → CCP (NPL)
│   └── Large / continuous state space?
│       ├── Need valid standard errors? → NNES
│       ├── Need per-feature diagnostics? → TD-CCP
│       └── Need fastest possible estimation? → SEES
│
└── NO (inverse RL — recover reward from demonstrations)
    ├── Know what features matter?
    │   ├── Need interpretable linear weights + SEs? → MCE-IRL
    │   ├── Need nonlinear reward? → MCE-IRL (Deep)
    │   └── Need reward that transfers across environments? → AIRL
    ├── Don't know what features matter? → f-IRL
    └── Just need a baseline? → BC
```

---

## The 9 Estimators

### 1. NFXP-NK — Structural MLE Gold Standard

**Papers**: Rust (1987); Iskhakov, Jorgensen, Rust & Schjerning (2016)

**Core theorem**: Maximizes the exact log-likelihood `ℓ(θ) = Σ log P(a|s;θ)` where choice probabilities come from solving the Bellman equation to machine precision at each outer step. Iskhakov et al.'s SA→NK polyalgorithm switches from successive approximation (globally convergent) to Newton-Kantorovich (quadratic convergence) near the fixed point. Analytical gradients via the implicit function theorem: the Frechet derivative `(I - β·P_π)⁻¹` differentiates through the Bellman fixed point without finite differences or backprop.

**Why it's irreplaceable**: The only estimator delivering **statistically efficient MLE with analytical standard errors**. BHHH optimization guarantees positive-definite Hessian from per-observation score outer products. This is the method where you can publish parameter estimates with proper confidence intervals, likelihood ratio tests, and information criteria.

**When to use**: Publication-grade structural estimates on tabular problems. Replication of canonical DDC results (Rust bus, occupational choice).

**Limitation**: O(|S|² × inner_iterations) per outer step. Intractable for |S| > ~10K or β > 0.995.

---

### 2. CCP — Fast Structural via Hotz-Miller Inversion

**Papers**: Hotz & Miller (1993); Aguirregabiria & Mira (2002, Econometrica)

**Core theorem (Hotz-Miller Inversion Lemma)**: Under additive separability and IID private shocks, observed choice probabilities P(a|s) uniquely invert to value differences: `v(s,a) - v(s,a') = e(s,a) - e(s,a')` where the emax correction `e(s,a) = γ_Euler - log P(a|s)` depends only on the shock distribution, not transitions or discount factor.

**Aguirregabiria-Mira NPL**: Iterating Hotz-Miller in CCP space converges to MLE when the equilibrium is Lyapunov stable (spectral radius of best-response Jacobian < 1). The **zero Jacobian property** at the NPL fixed point ensures first-order insensitivity to CCP estimation error.

**Why it's irreplaceable**: Completely avoids the Bellman inner loop. Value recovery is a single matrix inversion `(I - β·F_π)⁻¹` — O(|S|³) once versus thousands of iterations. K=1 gives instant consistent estimates; K=5-10 recovers MLE efficiency. The only method that works for **dynamic games** where computing Nash equilibria is infeasible.

**When to use**: Rapid specification search. Dynamic games (oligopoly entry/exit). Any setting where the inner-loop cost of NFXP is the bottleneck.

**Limitation**: Still requires full transition matrix. O(|S|³) inversion. Hotz-Miller K=1 is consistent but inefficient.

---

### 3. MCE-IRL — The Bridge Between Econ and ML

**Papers**: Ziebart (2010, PhD thesis — Theorems 1-3); Wulfmeier et al. (2016) for Deep variant

**Core theorem (Ziebart Theorem 1)**: The distribution maximizing **causal entropy** `H(A^T | S^T) = Σ_t H(A_t | S_{1:t}, A_{1:t-1})` subject to feature matching `E_π[φ] = E_data[φ]` has a recursive closed-form solution that is exactly the softmax Bellman equation: `V(s) = log Σ_a exp(θ·φ(s,a) + β·E[V(s')])`. The gradient is the feature matching residual: `∇θ = E_data[φ] - E_π[φ]`.

**Ziebart Theorem 3**: MCE minimizes worst-case prediction log-loss — the recovered policy is maximally robust to distribution shift while matching observed features.

**Critical distinction**: Causal entropy conditions each action only on information available at decision time `H(A_t | S_{1:t})`, not on future states. This is why MCE-IRL dominates MaxEnt IRL — it respects the information structure of sequential decisions, which is exactly what economists mean by "rational expectations."

**Why it's irreplaceable**: The only inverse estimator recovering **interpretable linear reward parameters with bootstrap standard errors** and a **provable robustness guarantee**. The softmax Bellman is simultaneously the MCE solution and the DDC choice model — an economist sees "structural logit with feature matching," an ML researcher sees "maximum entropy IRL." The Deep variant extends to nonlinear rewards via neural networks.

**When to use**: Recovering reward weights from demonstrations (route choice, labor decisions). Entry point for any IRL practitioner. Deep variant for complex nonlinear preferences.

**Limitation**: Requires known transitions and finite state space. Deep variant loses standard errors and interpretability.

---

### 4. TD-CCP — Neural Approximate VI with Feature Decomposition

**Paper**: Adusumilli & Eckardt (2025)

**Core theorem**: TD learning with averaging converges to the true value function at a rate sufficient for **√n-consistency of θ**. The key innovation is **per-feature EV decomposition**: instead of a monolithic V(s), TD-CCP learns K+1 separate neural networks — one per utility feature `EV_k(s)` plus an entropy network — each satisfying `EV_k(s) = Σ_a P(a|s)·φ_k(s,a) + β·E[EV_k(s')]`.

**Why it's irreplaceable**: The feature decomposition provides **interpretable per-component diagnostics** no other neural method offers. If EV₃ has high loss but EV₁ converges, you know the third feature's continuation value structure is complex. The CCP structure (frequency-based initial policy, NPL-style iteration) grounds it in econometric tradition while neural AVI enables scaling to continuous states.

**When to use**: Continuous state variables (experience, health capital, wealth) where discretization introduces massive approximation error and you need to understand which value function components drive decisions.

**Limitation**: Trains K+1 networks (expensive for large K). Semi-gradient TD lacks global convergence guarantees.

---

### 5. NNES — Neural V-Network with Valid Inference

**Paper**: Nguyen (2025, Georgetown JMP)

**Core theorems**:
- **Proposition 1 (Zero Jacobian)**: At the true CCP, `∂φ_θ₀(P*, V*)/∂P = 0`. Neural network errors in V have second-order effects on the likelihood.
- **Propositions 3-4 (Neyman Orthogonality)**: The likelihood score is orthogonal to V-approximation error. θ is √n-consistent even when V converges at the slower rate o_p(n^{-1/4}).
- **Theorem 4.3 (Semiparametric Efficiency)**: `√n(θ̂ - θ₀) → N(0, Σ⁻¹)` — achieves the efficiency bound with no bias correction.

**Why it's irreplaceable**: The only neural method where **standard errors are theoretically valid**. NFXP has valid SEs but can't scale. TD-CCP and SEES scale but their SEs are heuristic. NNES proves that the DDC likelihood's orthogonality structure insulates structural parameters from the nuisance V-network error.

**When to use**: High-dimensional continuous states where you need both neural scalability AND publication-grade inference (standard errors, hypothesis tests).

**Limitation**: Quality depends on initial parameter guess θ₀. Bellman residual ≠ 0 in practice.

---

### 6. SEES — Sieve Basis: The Fastest Scalable Estimator

**Paper**: Luo & Sang (2024); Arcidiacono et al. (2013)

**Core theorems**:
- **Theorem 1**: `sup_θ ||p̂(θ) - p*(θ)||₂ = O_p(1/√ω)` where ω is the penalization parameter.
- **Theorem 3**: `√n(θ̂ - θ₀) → N(0, Σ)` achieving the Cramer-Rao efficiency bound.

V(s) is approximated as `Ψ(s)·α` (Fourier or Chebyshev basis). Joint optimization of `[θ, α]` with L2 penalty on α — gradually enforcing model consistency without ever solving a fixed point.

**Why it's irreplaceable**: **Zero neural network training.** No SGD, no mini-batches, no learning rates. The basis matrix is deterministic; continuation values are a single matrix multiply `P_a @ Ψ`. The entire estimation is one L-BFGS-B call over ~10-20 parameters. Cost is **O(basis_dim), independent of |S|** — scales to |S| > 100K where neural methods hit memory limits.

**When to use**: Massive state spaces where neural methods are too slow and tabular methods are impossible. Quick estimation for model exploration. Settings where V is known to be smooth.

**Limitation**: Basis projection introduces systematic bias if V has sharp discontinuities. Basis selection is a modeling choice.

---

### 7. AIRL — Adversarial Reward Recovery with Transfer Guarantees

**Paper**: Fu, Luo & Levine (2018, ICLR)

**Core theorems**:
- **Theorem 5.1 (Disentanglement)**: A state-only reward r'(s) recovered by IRL is disentangled with respect to all dynamics — it transfers across environments.
- **Theorem 5.2 (Necessity)**: If a reward is disentangled for all dynamics, it must be state-only.

The discriminator structure `D = exp(f)/(exp(f) + π(a|s))` with `f = g_θ(s) + γh(s') - h(s)` forces the learned reward to be state-only by canceling potential-based shaping at optimality.

**Why it's irreplaceable**: The only IRL method with a **proven transfer guarantee**. GAIL matches occupancy measures but recovers rewards confounded with the value function. MCE-IRL recovers clean rewards but only under fixed dynamics. AIRL's disentanglement theorems show its learned reward generalizes when the environment changes.

**When to use**: Sim-to-real transfer. Learning rewards in one environment for deployment in another. Autonomous vehicles, robotics — anywhere training and deployment dynamics differ.

**Limitation**: Solves full MDP at each adversarial round. Adversarial training can be unstable. Slowest of the IRL methods.

---

### 8. f-IRL — Feature-Free Distribution Matching

**Paper**: Ni, Sikchi, Wang & Bhatt (2022, CoRL)

**Core principle**: Minimizes the f-divergence between state-action occupancy measures: `min_R D_f(ρ_expert || ρ_π)`. Gradient on tabular reward: KL → `log(ρ_E/ρ_π)`, χ² → `(ρ_E/ρ_π) - 1`, TV → `sign(ρ_E - ρ_π)`.

**Why it's irreplaceable**: The only IRL method requiring **zero assumptions about reward structure**. No features to design, no neural architecture to choose, no discriminator to train. It operates directly on occupancy measure mismatch. The choice of f-divergence gives a **menu of robustness properties**: KL for maximum likelihood equivalence, TV for robustness to outlier demonstrations, χ² for variance sensitivity.

**When to use**: You have demonstrations but no idea what features matter. Exploratory reward recovery where you inspect the tabular reward post-hoc to discover structure. Sensitivity analysis across divergence objectives.

**Limitation**: Tabular reward only. Horizon-truncated state visitation may underestimate long-run effects for high β.

---

### 9. BC — The Honest Baseline

**Papers**: Pomerleau (1991); Ross, Gordon & Bagnell (2011)

**Core result (Ross et al. 2011)**: BC's error compounds quadratically under distribution shift: `Error = O(T²ε)`. IRL methods that recover the true reward achieve O(ε) regardless of horizon.

**Why it's irreplaceable**: The only method using **zero MDP structure** — no transitions, no Bellman, no reward, no value function. Just `P̂(a|s) = N(s,a)/N(s)`. Establishes the floor: any method that can't beat BC hasn't learned from sequential decision-making structure.

**When to use**: Always run first. It tells you: (a) whether demonstrations contain enough signal, (b) whether your IRL method is leveraging MDP structure or memorizing frequencies, (c) a calibration target for policy evaluation.

**Limitation**: No reward recovery, no transfer, no generalization, no counterfactuals.

---

## Capability Matrix

Each row has at least one unique cell:

|  | Efficient MLE | Avoids Bellman | Inverse (demo→reward) | Valid Neural SEs | O(1) in \|S\| | Transfer | Feature-Free | Per-Feature Diagnostics | Zero MDP Structure |
|--|---|---|---|---|---|---|---|---|---|
| **NFXP** | **Yes** | | | | | | | | |
| **CCP** | | **Yes** | | | | | | | |
| **MCE-IRL** | | | **Yes + interpretable** | | | | | | |
| **TD-CCP** | | | | | | | | **Yes** | |
| **NNES** | | | | **Yes** | | | | | |
| **SEES** | | | | | **Yes** | | | | |
| **AIRL** | | | | | | **Yes** | | | |
| **f-IRL** | | | | | | | **Yes** | | |
| **BC** | | | | | | | | | **Yes** |
