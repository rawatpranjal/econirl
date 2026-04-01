# f-IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ni et al. (2022) | Tabular | Yes | No | No |

## Background

Every other IRL method asks you to choose something about the reward: features (MCE-IRL), a network architecture (deep MCE-IRL), or a discriminator design (AIRL). Ni et al. (2022) asked: can we do IRL with no assumptions at all? Their approach assigns a free reward parameter to every state-action pair and adjusts these parameters to make the model's behavior match the expert's as closely as possible, measured by an f-divergence. You get to choose the divergence: KL for a maximum-likelihood flavor, total variation for robustness to outliers, or chi-squared for sensitivity to distribution differences.

## Key Equations

$$
\min_r D_f(\rho_E \| \rho_{\pi_r}),
$$

where $\rho(s,a) = E_\pi\{\sum_{t=0}^\infty \beta^t \mathbb{I}(s_t = s, a_t = a)\}$ is the discounted occupancy measure. The gradient on the tabular reward depends on the divergence,

$$
\nabla_r = \begin{cases} \log(\rho_E / \rho_\pi) & \text{KL} \\ (\rho_E / \rho_\pi) - 1 & \chi^2 \\ \mathrm{sign}(\rho_E - \rho_\pi) & \text{TV} \end{cases}.
$$

## Pseudocode

```
f-IRL(D_expert, p, beta, sigma, divergence, lr):
  1. Estimate how often the expert visits each (s,a)
  2. Initialize reward r(s,a) = 0 everywhere
  3. Repeat:
     a. Solve for optimal policy under current r
     b. Compute how often this policy visits each (s,a)
     c. Adjust r to close the gap (gradient depends on chosen divergence)
  4. Return r (one number per state-action pair)
```

## Strengths and Limitations

f-IRL requires absolutely zero reward assumptions. It eliminates feature engineering and discriminator instability by working directly with occupancy measure matching. The choice of divergence gives flexibility: KL for likelihood-based reasoning, total variation for robustness, chi-squared for sensitivity analysis.

The limitation is that f-IRL is confined to purely tabular state-action spaces. It requires robust density in expert observations across the full state-action space to work well. Without enough data in every cell, the tabular reward estimates become noisy. It also produces no standard errors and no interpretable parameters.

f-IRL is the right choice for exploratory analysis when you have no idea which features matter and want to see what the raw reward landscape looks like before committing to a parametric form.

## References

- Ni, T., Sikchi, H., Wang, Y., Gupta, T., Lee, L., & Eysenbach, B. (2022). f-IRL: Inverse Reinforcement Learning via State Marginal Matching. *CoRL 2022*.
