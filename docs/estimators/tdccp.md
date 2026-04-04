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

## Diagnostics and Guarantees

Identification requires the same linear-in-parameters utility structure as other CCP-based methods, but TD-CCP does not require a known transition matrix. The method learns the expected value components directly from observed transitions using semi-gradient TD, so it works in settings where only trajectory data is available. Each feature component is learned by a separate network, and the entropy component is learned independently. If any component fails to converge, the per-component loss history reveals exactly which part of the value decomposition is unreliable.

The estimator has three levels of convergence. The innermost level trains each EV component network for a fixed number of approximate value iteration rounds (default 20), with 30 SGD epochs per round. The second level maximizes the partial pseudo-likelihood via L-BFGS-B with a gradient tolerance of 1e-6 and a maximum of 200 iterations. The outermost level runs policy iterations (default 3) that update the CCPs from the estimated parameters, with convergence declared when the parameter change falls below 1e-4.

Standard errors are computed from the numerical Hessian of the pseudo-likelihood evaluated at the optimum. The validity of these standard errors depends on the quality of the neural EV approximation. Unlike NNES, TD-CCP does not have a formal Neyman orthogonality guarantee, so large V-approximation errors can bias the Hessian. In practice, monitoring the per-component training losses helps assess whether the standard errors are trustworthy.

The main limitation is resource cost. The method trains K plus 1 separate neural networks, where K is the number of features and the additional network handles the entropy component. TD learning also carries no strict global convergence guarantee, so results should be cross-checked against other estimators when possible. The semi-gradient TD update uses stop-gradient on the bootstrap target, which can lead to slow convergence in environments with long time horizons.

The default configuration uses component networks with 2 hidden layers of 64 units each, a learning rate of 1e-3, mini-batch size of 8192, CCP smoothing of 0.01, 20 AVI rounds with 30 epochs per round, and 3 policy iterations. The L-BFGS-B optimizer for the structural MLE step uses a gradient tolerance of 1e-6 with a maximum of 200 iterations.
