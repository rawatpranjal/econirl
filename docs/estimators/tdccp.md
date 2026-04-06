# TD-CCP

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Adusumilli and Eckardt (2025) | Linear | No | Robust sandwich | Yes |

## What this estimator does

CCP estimation avoids the inner Bellman loop of NFXP by inverting observed choice probabilities into value differences, but the inversion requires the transition density $p(s'|s,a)$. When the state space is continuous or high-dimensional, estimating this density is the binding constraint. Adusumilli and Eckardt (2025) eliminate the transition density entirely. Their key insight is that the CCP value function components $h_k(a,x)$ and $g(a,x)$ satisfy temporal-difference fixed-point equations that depend only on observed transitions $(x_t, a_t, x_{t+1})$, not on the density itself. This is the same idea that makes TD learning in reinforcement learning model-free.

The estimator comes in two variants. The linear semi-gradient method approximates $h$ and $g$ with polynomial basis functions and solves a single linear system in sub-second time. The neural approximate value iteration method uses neural networks for $h$ and $g$, iterating Bellman-like regression targets until convergence. Both produce locally robust standard errors at the parametric $\sqrt{n}$ rate, even though the first-stage estimates converge at slower nonparametric rates, through Neyman orthogonality and 2-fold cross-fitting.

TD-CCP is the only estimator in this package that requires neither transition densities nor an inner Bellman loop while still providing valid standard errors.

## How it works

The CCP decomposition writes the choice-specific value as $Q(a,x;\theta) = \sum_k \theta_k h_k(a,x) + \sigma g(a,x)$. The semi-gradient method solves for each $h_k$ by inverting

$$
\hat\omega_k = \left[\frac{1}{n}\sum_{i} \Psi_i(\Psi_i - \beta\Psi_i')^\top\right]^{-1} \frac{1}{n}\sum_{i} \Psi_i z_{k,i},
$$

where $\Psi_i$ and $\Psi_i'$ are basis evaluations at the current and next transitions. This is a single matrix inversion shared across all $K$ feature components. The structural parameters $\theta$ are then estimated by pseudo-maximum likelihood on a held-out fold. Locally robust standard errors correct for first-stage estimation error using backward value functions that solve an adjoint equation.

## When to use it

TD-CCP is the right choice when transition densities are unavailable or too expensive to estimate, which is the typical situation with continuous state spaces. On tabular problems where transitions are known, NFXP and CCP are slightly more precise because they use exact continuation values. TD-CCP's unique contribution is combining transition-free estimation with valid $\sqrt{n}$ inference. For counterfactual analysis after estimation, the transition density is needed to re-solve the Bellman equation, but it can be specified as a modeling assumption rather than estimated from the data used for parameter recovery.

## References

- Adusumilli, K. and Eckardt, D. (2025). Temporal-Difference Estimation of Dynamic Discrete Choice Models. Working paper.

The full derivation, algorithm, and simulation results are in the [TD-CCP primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/tdccp.pdf).
