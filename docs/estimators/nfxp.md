# NFXP

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Rust (1987), Iskhakov et al. (2016) | Linear | Yes | Analytical (MLE) | No |

## What this estimator does

NFXP is the foundational structural estimator for dynamic discrete choice models. Rust (1987) proposed nesting the Bellman fixed point inside maximum likelihood. For every candidate parameter vector $\theta$, the inner loop solves the Bellman equation to machine precision, producing the softmax choice probabilities that enter the log-likelihood. The outer loop searches over $\theta$ to maximize fit. Because the inner loop is exact, the likelihood surface is smooth and the estimator achieves full statistical efficiency.

The bottleneck is the inner loop. Value iteration converges at rate $\beta$ per step, so patient agents with $\beta$ near 1 require thousands of contractions per likelihood evaluation. Iskhakov, Rust and Schjerning (2016) solved this with the SA-then-NK polyalgorithm. The algorithm starts with successive approximation for safe global convergence and switches to Newton-Kantorovich near the fixed point for quadratic convergence. The total iteration count becomes insensitive to $\beta$.

Identification requires that the feature matrix have full column rank, which means features must vary across actions. State-only features that are identical for all actions collapse the likelihood surface. The discount factor $\beta$ is not identified without a priori restrictions and must be fixed by the researcher. The scale parameter $\sigma$ is normalized to 1.

## How it works

The estimator maximizes

$$
\hat\theta_{\mathrm{NFXP}} = \arg\max_\theta \sum_{i=1}^{N} \log \pi(a_i \mid s_i; \theta),
$$

where $\pi(a \mid s; \theta)$ is the softmax policy from solving $V = TV$ at each $\theta$. Gradients are computed analytically through the fixed point using the implicit function theorem. The per-observation score solves a linear system $(I - \beta P_\pi) \frac{\partial V}{\partial \theta} = \sum_a \pi(a|s) \phi(s,a)$ and enters the BHHH outer product to form the Hessian approximation. Standard errors come directly from the inverse of this outer product, with no bootstrap required.

## When to use it

NFXP is the right choice when you need publication-grade structural estimates with analytical standard errors, likelihood ratio tests, and hypothesis tests on a manageable state space. The inner loop costs $O(|\mathcal{S}|^2)$ per contraction and the NK step costs $O(|\mathcal{S}|^3)$, so state spaces above a few thousand become slow. For large or continuous state spaces, consider NNES, SEES, or TD-CCP instead. Every other estimator in this package defines itself relative to NFXP.

## References

- Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999-1033.
- Iskhakov, F., Lee, J., Rust, J., Schjerning, B., & Seo, K. (2016). Comment on "Constrained Optimization Approaches to Estimation of Structural Models." *Econometrica*, 84(1), 365-370.

The full derivation, algorithm, and simulation results are in the [NFXP primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/nfxp.pdf).
