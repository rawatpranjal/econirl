# GLADIUS

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Model-Free Neural | Kang et al. (2025) | Neural (projects onto linear) | No | No (R² diagnostic) | Yes |

## Background

Some previous methods (GAIL, IQ-Learn) only match behavior on states the expert actually visited, leaving the value function undefined elsewhere. Others (NFXP, CCP) need the transition matrix or a big matrix inversion. Kang et al. (2025) built GLADIUS to avoid both problems. It uses two neural networks: one for $Q(s,a)$ and one for the expected next-period value $EV(s,a)$. The key insight is that the standard training objective (TD error) mixes together Bellman error (what we care about) and transition noise (which is irreducible). By using the second network to estimate the noise, GLADIUS isolates the true Bellman error. It needs no transition model. After training, the reward is read off as $r = Q - \beta \cdot EV$, and structural parameters come from projecting this reward onto features.

## Key Equations

$$
\max_{\phi_1} \bigg[\ell^p(\phi_1) - \lambda \, \rho_{BE}(Q_{\phi_1})\bigg].
$$

The mean squared TD error decomposes into Bellman error and irreducible transition variance,

$$
\rho_{TD}(Q) = \rho_{BE}(Q) + \beta^2 E_p\big\{[V(s') - EV(s,a)]^2\big\}.
$$

After training, the reward is recovered from the Bellman identity,

$$
r(s,a) = Q_{\hat\phi_1}(s,a) - \beta \, EV_{\hat\phi_2}(s,a).
$$

## Pseudocode

```
GLADIUS(D, features, beta, sigma, lambda):
  1. Initialize Q-network and EV-network
  2. For each training epoch:
     a. Sample a mini-batch of (s, a, s') from data
     b. Compute V(s') from Q-network via log-sum-exp
     c. Loss = choice prediction error + lambda * Bellman error
     d. Update Q-network to minimize loss
     e. Update EV-network to track expected V
  3. Recover reward: r(s,a) = Q(s,a) - beta * EV(s,a)
  4. Project reward onto features: theta = least-squares fit
  5. R-squared tells you how linear the learned reward is
  6. Return theta, r, R-squared
```

## Strengths and Limitations

GLADIUS avoids both the transition matrix requirement of structural estimators and the on-support limitation of imitation learning methods. The dual-network architecture cleanly separates Bellman error from irreducible transition noise, giving a principled training objective. After training, the R-squared from projecting the neural reward onto linear features tells you how much of the learned reward your parametric model explains.

The limitation is that GLADIUS does not produce standard errors on the structural parameters. The least-squares projection from neural reward to $\theta$ is a post-hoc step, not a formal estimator with known asymptotic properties. Training also requires tuning the penalty weight $\lambda$ and the network architectures for both the Q-network and the EV-network.

GLADIUS is the right choice when you want model-free estimation with a diagnostic on how well your linear features capture the true reward.

## References

- Kang, H. et al. (2025). GLADIUS: Model-Free Estimation of Dynamic Discrete Choice Models.
