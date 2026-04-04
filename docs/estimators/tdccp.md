# TD-CCP

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Adusumilli and Eckardt (2025) | Linear | No (uses data) | Robust sandwich | Yes (semi-gradient or neural AVI) |

## Background

Adusumilli and Eckardt (2025) develop TD-CCP as a general-purpose CCP estimator for continuous and high-dimensional state spaces. The standard CCP approach of Hotz and Miller (1993) requires estimating the transition density K(x'|a,x) to compute the recursive value function terms h(a,x) and g(a,x) that appear in the pseudo-log-likelihood. When the state space is continuous, estimating K is difficult and introduces bias. TD-CCP avoids this entirely by learning h and g directly from observed (a, x, a', x') transition tuples using temporal-difference methods from reinforcement learning. No transition density estimation is required.

The estimator offers two methods. The linear semi-gradient method approximates h and g using basis functions and solves for the fixed point in closed form via a single matrix inversion. This is fast (sub-second for moderate problems) and is the recommended default. The approximate value iteration (AVI) method iteratively trains function approximators (neural networks, random forests, or any ML method) to regress on TD targets. AVI is more flexible for very high-dimensional state spaces.

For valid inference, the paper introduces a locally robust moment condition that accounts for the nonparametric estimation error in h and g. This correction uses a backward value function and 2-fold cross-fitting to achieve parametric convergence rates. Without these corrections, standard errors understate uncertainty.

## Key Equations

The pseudo-log-likelihood from equation (2.1) is maximized over the structural parameters theta. The recursive terms h(a,x) and g(a,x) satisfy Bellman-like equations (2.2) that TD methods can solve from data.

The linear semi-gradient fixed point (equation 3.5) is

$$
\hat\omega = \Big[\mathbb{E}_n\big[\phi(a,x)\big(\phi(a,x) - \beta \phi(a',x')\big)^\top\big]\Big]^{-1} \mathbb{E}_n\big[\phi(a,x) \, z(a,x)\big]
$$

where phi(a,x) are basis functions over action-state pairs, z(a,x) are the utility features, and (a, x, a', x') are observed transitions. The estimated h is then phi(a,x)^T omega for each component.

The AVI update (equation 3.10) iteratively solves

$$
\hat{h}_{j+1} = \arg\min_{f \in \mathcal{F}} \, \mathbb{E}_n\Big[\big(z(a,x) + \beta\,\hat{h}_j(a',x') - f(a,x)\big)^2\Big]
$$

Each iteration is a standard regression problem. The target is the known reward z(a,x) plus the discounted previous estimate evaluated at the observed next state-action pair.

## Pseudocode

```
TD-CCP(D, features, beta, sigma):
  1. Estimate CCPs: P_hat(a|s) via frequency counting or logit
  2. Extract observed transitions (a, x, a', x') from panel data
  3. Estimate h(a,x) and g(a,x):
     Semi-gradient: solve omega = A^{-1} b (one matrix inversion)
     Neural AVI: iteratively train h_{j+1} on targets z + beta * h_j(a', x')
  4. Maximize pseudo-log-likelihood Q(theta; h_hat, g_hat) via L-BFGS-B
  5. (Cross-fitting) Repeat steps 3-4 on each data fold, average theta
  6. Compute locally robust SEs via backward value function and sandwich formula
  7. Return theta, robust SEs
```

## Strengths and Limitations

TD-CCP is model-free in the sense that it never estimates or uses the transition density. This is its main advantage over NFXP and CCP, which require a transition matrix. The linear semi-gradient method is extremely fast and gives a closed-form solution. The per-feature h decomposition provides interpretable diagnostics, showing which features contribute most to the continuation value.

The locally robust standard errors are a major advance. Unlike the naive Hessian-based SEs used by most other neural estimators, the paper's debiased moment condition produces valid confidence intervals at parametric rates even when the value function is estimated nonparametrically. This makes TD-CCP one of the few neural-scale estimators with theoretically valid inference.

The limitation of the linear semi-gradient method is basis function selection. Poor basis functions lead to approximation bias in h and g, which propagates to theta. The paper recommends polynomial bases of degree 2 to 4 and provides a cross-validation procedure for selection. The neural AVI method avoids this choice but is slower and carries no global convergence guarantee, so results should be cross-checked against the semi-gradient or other estimators.

TD-CCP is the right choice when the state space is continuous or high-dimensional, the transition density is unknown or hard to estimate, and you need valid standard errors. For tabular problems where transitions are available, NFXP or CCP may be simpler.

## References

- Adusumilli, K. and Eckardt, D. (2025). "Temporal-Difference Estimation of Dynamic Discrete Choice Models."
- Hotz, V. J. and Miller, R. A. (1993). "Conditional Choice Probabilities and the Estimation of Dynamic Models." Review of Economic Studies 60(3), 497-529.
- Tsitsiklis, J. N. and Van Roy, B. (1997). "An Analysis of Temporal-Difference Learning with Function Approximation." IEEE TAC 42(5).

## Diagnostics and Guarantees

Identification requires the same linear-in-parameters utility structure as other CCP-based methods. The key additional requirement is that the basis functions (for semi-gradient) or function class (for AVI) can approximate h and g well enough. Theorem 1 of the paper bounds the approximation error as (1-beta)^{-1} times the basis truncation error, so higher discount factors demand more flexible approximations.

The semi-gradient method converges in a single step because it computes the fixed point of the projected Bellman operator directly. No iteration is needed. The AVI method converges when the relative change ratio falls below 0.01 (footnote 9 of the paper), which typically happens within 20 iterations. The partial MLE outer loop uses L-BFGS-B with a gradient tolerance of 1e-6 and a maximum of 200 iterations.

Standard errors are computed via the locally robust sandwich formula from Section 4 of the paper when cross-fitting is enabled. The backward value function lambda is estimated using the same semi-gradient method applied to the backward Bellman operator. The sandwich variance is V = (G^T Omega^{-1} G)^{-1} where G is the Jacobian of the robust moment and Omega is its variance. When cross-fitting is disabled, the estimator falls back to the naive numerical Hessian, which does not account for first-stage estimation error and produces standard errors that are too small.

The default configuration uses the semi-gradient method with polynomial basis functions of dimension 8, 2-fold cross-fitting, and locally robust standard errors. The CCP first stage uses frequency counting with smoothing of 0.01. Policy iteration is set to 1 (matching the paper), meaning h and g are estimated once and plugged directly into the pseudo-log-likelihood. Setting policy iterations above 1 adds NPL refinement, which re-solves the Bellman equation and is not part of the paper's algorithm.
