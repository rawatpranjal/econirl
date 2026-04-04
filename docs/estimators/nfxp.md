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

## Diagnostics and Guarantees

Parameters are identified under standard regularity conditions for maximum likelihood in discrete choice models. The utility features must vary across actions for the likelihood surface to be well-defined. State-only features that are identical across all choices collapse the choice probabilities to a constant, making the likelihood flat and causing parameter blowup. The transition matrix must have full support on the reachable state space.

The estimator uses dual convergence criteria. The inner loop solves the Bellman equation to a tolerance of 1e-12 by default, using the SA-then-NK polyalgorithm from Iskhakov et al. (2016) that starts with successive approximation and switches to Newton-Kantorovich once the error drops below 1e-3. The outer loop terminates when the maximum absolute gradient falls below 1e-6 or when the log-likelihood change is less than 1e-10 for more than ten iterations. The default outer optimizer is BHHH with a maximum of 1000 iterations.

Standard errors come from the observed information matrix. In BHHH mode, the estimator computes per-observation scores via the implicit function theorem and forms the outer product of gradients. In L-BFGS-B or BFGS mode, the Hessian is computed numerically at the optimum. Both approaches produce valid asymptotic standard errors because NFXP is a full maximum likelihood estimator with an exact inner loop.

The main practical limitation is computational cost. Solving the Bellman equation costs O(S squared) per outer iteration, so state spaces above roughly 10,000 become slow. Discount factors above 0.995 also hurt because the contraction rate approaches one. The inner solver warns when the discount factor exceeds 0.99 and the maximum iteration count is below 50,000. Feature normalization matters for conditioning. The Rust bus parameters of 0.001 and 3.0 span three orders of magnitude, so rescaling features to a common range improves optimizer behavior.

The default configuration uses the hybrid inner solver with a switch tolerance of 1e-3, inner tolerance of 1e-12, and a maximum of 100,000 inner iterations. The outer loop uses BHHH with a gradient tolerance of 1e-6 and a maximum of 1000 iterations. These defaults are designed for problems with moderate discount factors and state spaces up to a few thousand states.
