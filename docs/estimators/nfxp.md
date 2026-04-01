# NFXP

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Rust (1987), Iskhakov et al. (2016) | Linear | Yes | Analytical (MLE) | No |

## Background

To evaluate how well any candidate $\theta$ fits the data, we need the choice probabilities $\pi_\theta$. But computing $\pi_\theta$ requires solving the Bellman equation, which is itself an expensive fixed-point problem. Rust (1987) handled this by nesting the Bellman solve inside the optimizer. For every trial $\theta$, the inner loop solves for $Q_\theta$, the outer loop checks how well the resulting $\pi_\theta$ fits the data, and then tries a better $\theta$. This is expensive but gives exact maximum likelihood with proper standard errors. Iskhakov et al. (2016) sped up the inner loop with a two-phase approach: start with simple iteration (safe but slow), then switch to Newton's method (fast but needs a good starting point) once you are close to the answer.

## Key Equations

$$
\hat\theta_{\mathrm{NFXP}} = \arg\max_\theta \sum_{(s,a) \in \mathcal{D}} \log \pi^*_\theta(a \mid s),
$$

where $\pi^*_\theta$ comes from solving $Q_\theta = \Lambda_\sigma(Q_\theta)$ for each candidate $\theta$. Gradients are computed analytically through the fixed point using the implicit function theorem.

## Pseudocode

```
NFXP(D, r_theta, p, beta, sigma):
  1. Initialize theta
  2. Repeat until convergence:
     a. Inner loop — solve for Q_theta:
        Q <- 0
        Repeat (simple iteration):
          Q <- Lambda_sigma(Q; theta)
        Until close enough, then switch to Newton:
          Q <- Q - (I - beta*P_pi)^{-1} (Q - Lambda_sigma(Q))
        Until converged
     b. Compute policy: pi(a|s) = softmax(Q(s,.)/sigma)
     c. Compute log-likelihood: L = sum log pi(a|s) over data
     d. Compute gradient analytically
     e. Update theta (BHHH step)
  3. Standard errors from inverse Hessian
  4. Return theta, SEs
```

## Strengths and Limitations

NFXP delivers statistically efficient maximum likelihood estimates with precise analytical standard errors. It guarantees a globally convergent, machine-precision solution. The inner loop converges to the exact fixed point, so no approximation error contaminates the parameter estimates.

The downside is cost. Solving the Bellman equation takes $O(|\mathcal{S}|^2)$ per outer optimization step, which becomes impractical when the state space exceeds about 10,000. Highly forward-looking settings with $\beta > 0.995$ also slow convergence because the contraction rate approaches one. For large or continuous state spaces, consider NNES or SEES instead.

NFXP is the right choice when you need publication-grade structural estimates with confidence intervals, likelihood ratio tests, and hypothesis tests on a manageable state space.

## References

- Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999-1033.
- Iskhakov, F., Lee, J., Rust, J., Schjerning, B., & Seo, K. (2016). Comment on "Constrained Optimization Approaches to Estimation of Structural Models." *Econometrica*, 84(1), 365-370.
