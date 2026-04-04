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

## Diagnostics and Guarantees

Q-values are identified only up to a state-dependent constant c(s) when rewards are not observed. NLL-only training pins the Q-value differences across actions at each state, but the absolute level floats freely. This constant leaks into the implied rewards r(s,a) through the transition structure asymmetrically, because different actions lead to different successor states. On the Rust bus problem with beta of 0.95, this structural bias causes GLADIUS to overestimate the operating cost parameter by roughly 40 percent while recovering the replacement cost within 8 percent. This bias persists regardless of network size, training duration, or data volume. When rewards are observed in the data, the bi-conjugate Bellman error anchors the Q-value scale and eliminates this bias.

The estimator uses early stopping as its convergence criterion. Training runs for up to 500 epochs by default, with convergence declared when the average loss does not improve by more than 1e-6 for 50 consecutive epochs (the patience parameter). If the maximum number of epochs is reached without early stopping, the estimator still reports convergence. The alternating update scheme from Algorithm 1 of Kang et al. (2025) updates the zeta-network on even batches and the Q-network on odd batches, with each network frozen while the other trains.

GLADIUS does not produce analytical standard errors on the structural parameters. The post-hoc least-squares projection from neural reward onto linear features is not a formal estimator with known asymptotic properties. The R-squared from this projection serves as a diagnostic, telling the user how much of the learned neural reward is captured by the linear feature specification. Bootstrap standard errors can be requested (default 100 replications), but their validity depends on the stability of the neural training across bootstrap samples.

The main practical limitations are the structural identification bias in the IRL setting (described above) and the sensitivity to hyperparameter tuning. The Bellman penalty weight lambda, the learning rates for both networks, and the network architectures all affect the quality of the recovered reward. The Q-network should not receive Bellman gradients through V_Q in the IRL setting, because this causes Q-value explosion without observed rewards to anchor the scale.

The default configuration uses Q and EV networks with 3 hidden layers of 128 units each, learning rates of 1e-3 for both networks, a batch size of 512, Bellman penalty weight of 1.0, weight decay of 1e-4, gradient clipping at 1.0, and a learning rate decay rate of 0.001. Early stopping uses a patience of 50 epochs out of a maximum of 500.
