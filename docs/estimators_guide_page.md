# Guide

This page helps you choose the right estimator.

## Quick Reference

| Estimator | Direction | Reward Type | Needs Transitions | Standard Errors | Scales Beyond Tabular | Transfer |
|-----------|-----------|-------------|-------------------|-----------------|----------------------|----------|
| NFXP-NK | Forward ($\theta \to \pi$) | Linear | Yes | Analytical (MLE) | No | No |
| CCP | Forward ($\theta \to \pi$) | Linear | Yes | Hessian | No | No |
| MCE-IRL | Inverse ($\pi \to \theta$) | Linear / Neural | Yes | Bootstrap | Deep variant only | No |
| TD-CCP | Forward ($\theta \to \pi$) | Linear | Yes | Hessian | Yes (neural AVI) | No |
| NNES | Forward ($\theta \to \pi$) | Linear | Yes | Valid (orthogonality) | Yes (neural V) | No |
| SEES | Forward ($\theta \to \pi$) | Linear | Yes | Marginal Hessian | Yes ($O(1)$ in $|\mathcal{S}|$) | No |
| AIRL | Inverse ($\pi \to R$) | Linear / Tabular | Yes | No | No | Yes |
| GLADIUS | Inverse ($\pi \to R$) | Linear (projected) | Yes | Projected | Yes (neural Q) | No |
| f-IRL | Inverse ($\pi \to R$) | Tabular | Yes | No | No | No |
| BC | Imitation | None | No | No | No | No |

## Identifying Assumptions

Every estimator must resolve the fundamental non-identification of rewards in dynamic discrete choice models. Observed choice probabilities identify only differences in Q-values across actions, not the level of rewards. Without further restrictions, any reward of the form $r'(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) h(s') - h(s)$ for an arbitrary function $h$ yields the same optimal policy (Ng et al. 1999). The estimators in this package use three distinct strategies to resolve this ambiguity.

| Estimator | Reward Form | Identifying Restriction | Recovers | Needs Known $p$ |
|-----------|-------------|------------------------|----------|---------------|
| NFXP | Linear $r = \theta \cdot \phi(s,a)$ | Parametric form with $\dim(\theta) < \lvert S \rvert(\lvert A \rvert-1)$ | $\theta$ (structural params) | Yes |
| CCP | Linear $r = \theta \cdot \phi(s,a)$ | Parametric form (same as NFXP) | $\theta$ (structural params) | Yes |
| MCE-IRL | Linear $r = \theta \cdot \phi(s,a)$ | Feature matching and norm constraint | $\theta$ (feature weights) | Yes |
| NNES | Linear $r = \theta \cdot \phi(s,a)$ | Parametric form and neural $V$ approximation | $\theta$ (structural params) | Yes |
| TD-CCP | Linear $r = \theta \cdot \phi(s,a)$ | Parametric form and TD approximation | $\theta$ (structural params) | No (uses data transitions) |
| SEES | Linear $r = \theta \cdot \phi(s,a)$ | Parametric form and sieve $V$ approximation | $\theta$ (structural params) | Yes |
| IQ-Learn | Nonparametric $R(s,a)$ | Chi-squared regularizer (min $\ell^2$ norm of implied reward) | $R(s,a)$ matrix | Yes (tabular) |
| GLADIUS | Neural $Q(s,a)$ projected to linear | Bi-conjugate Bellman error and linear projection | $\theta$ (projected structural params) | Yes |
| AIRL | Nonparametric $R(s)$ | Disentanglement (state-only reward and shaping potential) | $R(s)$ function | No (adversarial) |
| f-IRL | Nonparametric $R(s,a)$ | f-divergence minimization (occupancy matching) | $R(s,a)$ matrix | Yes |

The first six estimators (NFXP through SEES) assume a parametric reward $r(s,a) = \sum_k \theta_k \phi_k(s,a)$ where the features $\phi$ are known. This reduces the unknown rewards from $\lvert S \rvert \times \lvert A \rvert$ free values to a small number of structural parameters and is what makes direct comparison of parameter estimates possible. All six maximize the same conditional log-likelihood over the same parameters, differing only in how they solve the forward problem. NFXP uses a Bellman inner loop. CCP uses Hotz-Miller inversion. MCE-IRL uses feature matching, which is mathematically equivalent to the CCP likelihood under the AS/CI/EV assumptions. NNES and TD-CCP approximate the value function with neural networks. SEES uses sieve basis functions.

The last three estimators (IQ-Learn, AIRL, f-IRL) do not assume a parametric reward form. They recover a nonparametric reward function, one value per state-action pair, by backing out the implied reward from the learned Q-function via the inverse Bellman operator. Because the reward is not restricted to a parametric family, identification requires an alternative strategy. IQ-Learn uses the chi-squared regularizer to select the reward with smallest $\ell^2$ norm at expert-visited states. AIRL uses the discriminator structure to disentangle state-only rewards from the shaping potential. f-IRL minimizes an f-divergence between expert and learned occupancy measures.

Comparing estimators across these two groups requires evaluating on quantities that are identified under both approaches. Policy recovery and reward differences between actions are identified under all methods. Raw parameter estimates like $\theta_c$ are meaningful only for the parametric group.

## When to Use Each Estimator

**NFXP-NK** is for when you need the best possible estimates with reliable standard errors on a manageable state space. Think published papers, confidence intervals, hypothesis tests. Use it for classic problems like bus engine replacement or occupational choice.

**CCP** is for when NFXP is too slow. It skips the Bellman solve entirely. One step gives quick estimates, five to ten NPL steps match NFXP quality. It is the only option for dynamic games.

**SEES** is for big state spaces without neural networks. One optimization call, no training loops. Scales past 100,000 states.

**NNES** is for big state spaces when you need standard errors you can trust. It is the only neural method with theoretically valid inference.

**TD-CCP** is for continuous states when you want to see which features are easy or hard to predict. It works without knowing the transition probabilities.

**MCE-IRL** is for learning reward weights from demonstrations with standard errors. It is the bridge between IRL and structural econometrics.

**AIRL** is for when the learned reward needs to work in a different environment than the one it was trained in.

**GLADIUS** is for continuous state spaces when you want both neural flexibility and interpretable parameters. It learns a neural reward and projects it onto linear features. Check the projection R-squared to see how much the linear approximation captures.

**f-IRL** is for exploring what the reward looks like when you have no idea which features matter.

**BC** should always be your first step. If nothing beats it, the data may not have enough structure for model-based methods.

## Capability Matrix

Each row has at least one unique cell.

|  | Efficient MLE | Avoids Bellman | Inverse (demo to reward) | Valid Neural SEs | $O(1)$ in $\lvert\mathcal{S}\rvert$ | Transfer | Feature-Free | Per-Feature Diagnostics | Zero MDP Structure |
|--|---|---|---|---|---|---|---|---|---|
| **NFXP** | **Yes** | | | | | | | | |
| **CCP** | | **Yes** | | | | | | | |
| **MCE-IRL** | | | **Yes + interpretable** | | | | | | |
| **TD-CCP** | | | | | | | | **Yes** | |
| **NNES** | | | | **Yes** | | | | | |
| **SEES** | | | | | **Yes** | | | | |
| **AIRL** | | | | | | **Yes** | | | |
| **GLADIUS** | | | **Yes + projected** | | **Yes** | | | | |
| **f-IRL** | | | | | | | **Yes** | | |
| **BC** | | | | | | | | | **Yes** |
