# CCP and NPL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Hotz and Miller (1993), Aguirregabiria and Mira (2002) | Linear | Yes | Hessian | No |

## Background

Hotz and Miller (1993) found a shortcut. Under the Gumbel shock assumption, you can estimate choice probabilities directly from data (just count how often each action is chosen in each state), and then invert them to recover value differences without ever solving the Bellman equation. This replaces thousands of iterations with a single matrix inversion. The tradeoff is that the one-shot estimator is less precise than full MLE. Aguirregabiria and Mira (2002) showed you can get the precision back by iterating: estimate $\theta$, update the choice probabilities, re-estimate $\theta$, and repeat. They called this nested pseudo-likelihood (NPL). After a few rounds, the estimates match what NFXP would give.

## Key Equations

$$
\pi_\theta(a \mid s) = \frac{\exp\!\big(\hat{R}(s,a)^\top \theta + \hat{Q}_\varepsilon(s,a)\big)}{\sum_{a'} \exp\!\big(\hat{R}(s,a')^\top \theta + \hat{Q}_\varepsilon(s,a')\big)},
$$

where $\hat{R}$ and $\hat{Q}_\varepsilon$ come from a single matrix inversion $(I - \beta F_\pi)^{-1}$ using the empirical choice probabilities. The Hotz-Miller inversion recovers $Q$-value differences directly from observed choices,

$$
Q(s,a) - Q(s,a') = \sigma \log\!\big(\pi(a \mid s) / \pi(a' \mid s)\big).
$$

## Pseudocode

```
CCP(D, features, p, beta, sigma, K_npl):
  1. Count choices: pi_hat(a|s) = N(s,a) / N(s)
  2. For k = 1 to K_npl:
     a. One matrix inversion to get R and Q_eps
     b. Plug into logit formula to get pi_theta
     c. Maximize partial likelihood over theta
     d. Update pi_hat from new theta (NPL step)
  3. Standard errors from inverse Hessian
  4. Return theta, SEs
```

## Strengths and Limitations

CCP provides rapid structural estimation via a single matrix inversion, replacing the expensive Bellman inner loop of NFXP. One step ($K=1$) is fast and consistent. Five to ten NPL steps recover full MLE efficiency. This speed advantage allows for specification searching without hours of compute.

CCP is also the only option for dynamic games, where computing Nash equilibria through NFXP is otherwise intractable. The limitation is that it still requires a full transition matrix and bounded state spaces. The one-shot estimator ($K=1$) is statistically less efficient than full MLE, though NPL iteration closes this gap.

CCP is the right choice when NFXP is too slow, when you need to test many specifications quickly, or when estimating dynamic games.

## References

- Hotz, V. J. & Miller, R. A. (1993). Conditional Choice Probabilities and the Estimation of Dynamic Models. *Review of Economic Studies*, 60(3), 497-529.
- Aguirregabiria, V. & Mira, P. (2002). Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models. *Econometrica*, 70(4), 1519-1543.

## Diagnostics and Guarantees

Identification requires the same conditions as NFXP. The utility features must vary across actions, and the transition matrix must be known. The initial CCP estimates must be nondegenerate, meaning no action can have exactly zero probability in any state. The estimator applies Laplace smoothing of 1e-6 by default to prevent log-of-zero in the emax correction. States with fewer observations than the minimum count threshold (default 1) receive uniform choice probabilities.

Convergence has two levels. At the inner level, each NPL iteration maximizes a pseudo-likelihood via L-BFGS-B with a gradient tolerance of 1e-6 and a maximum of 1000 iterations. At the outer level, NPL convergence is declared when the Euclidean norm of the parameter change between iterations falls below 1e-6. With a single policy iteration (Hotz-Miller, the default), the estimator always reports convergence. With multiple NPL iterations, the estimator reports whether the outer loop converged within the specified number of rounds.

Standard errors are computed from the numerical Hessian of the full log-likelihood, not the pseudo-likelihood with fixed CCPs. The pseudo-likelihood Hessian can be rank-deficient in directions that primarily affect CCPs, so the estimator re-solves the Bellman equation at each perturbation to get the correct curvature. Per-observation gradient contributions are also computed for sandwich (robust) standard errors. Both Hessian-based and sandwich standard errors are valid after NPL convergence.

The one-shot Hotz-Miller estimator is consistent but statistically less efficient than full MLE. Five to ten NPL iterations typically close the efficiency gap. The matrix inversion step computes (I minus beta times F_pi) inverse, which has a condition number of roughly 1/(1 minus beta). At high discount factors above 0.99, the estimator automatically switches to float64 arithmetic to maintain numerical accuracy.

The default configuration uses one policy iteration (Hotz-Miller), a CCP smoothing constant of 1e-6, L-BFGS-B for the inner optimization with gradient tolerance 1e-6, and a maximum of 1000 optimization iterations per NPL step. Setting num_policy_iterations to 10 or higher is recommended when full MLE efficiency is needed.
