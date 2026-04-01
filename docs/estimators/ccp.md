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
