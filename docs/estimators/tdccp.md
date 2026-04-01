# TD-CCP

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Adusumilli and Eckardt (2025) | Linear | No (uses data) | Hessian | Yes (neural AVI) |

## Background

Adusumilli and Eckardt (2025) wanted to combine the transparency of CCP with the scalability of neural methods, and to do it without knowing the transition probabilities. Their trick was to learn each piece of the $Q$ decomposition separately using temporal difference (TD) learning on the raw data. For each feature $k$, they train a separate function $R_k$ on observed transitions $(s, a, s', a')$. This is model-free because it never uses $p(s' \mid s,a)$ directly. The per-feature breakdown also helps with debugging: if one component converges but another does not, you know exactly which part of the model is causing trouble.

## Key Equations

$$
\hat\phi = \Big[E_\mathcal{D}\big\{\nu(s,a)\big[\nu(s,a) - \beta \nu(s',a')\big]^\top\big\}\Big]^{-1} E_\mathcal{D}\big\{\nu(s,a) \, \vec{r}(s,a)\big\},
$$

where $\nu(s,a)$ are basis functions and $(s,a,s',a')$ are observed transitions.

## Pseudocode

```
TD-CCP(D, features, beta, sigma, basis_functions):
  1. Count choices: pi_hat(a|s) = N(s,a) / N(s)
  2. For each feature k:
     Learn R_k from data using TD regression
  3. Learn Q_eps similarly
  4. Plug R_hat and Q_eps_hat into the logit formula
  5. Maximize partial likelihood over theta
  6. Apply a bias correction for robustness
  7. Return theta, SEs
```

## Strengths and Limitations

TD-CCP is model-free and gives per-feature diagnostics. By isolating networks for specific features, researchers can debug exactly which component is failing to converge. This interpretability is unique among the scalable estimators. It also works without knowing the transition probabilities, using observed transitions directly.

The limitation is resource cost. The approach requires $K+1$ discrete networks (one per feature plus one for $Q_\varepsilon$), which grows with the number of features. TD learning also carries no strict global convergence guarantee, so results should be checked against other estimators when possible.

TD-CCP is the right choice for continuous states when you want to see which features are easy or hard to predict and you do not have access to transition probabilities.

## References

- Adusumilli, K. & Eckardt, D. (2025). Temporal Difference CCP Estimation.
