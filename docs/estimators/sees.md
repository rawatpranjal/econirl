# SEES

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Luo and Sang (2024) | Linear | Yes | Marginal Hessian | Yes, $O(1)$ in $|\mathcal{S}|$ |

## Background

NFXP and CCP both need to work with the full state space, either solving the Bellman equation or inverting a big matrix. When there are many states (large or continuous), this becomes impossible. Luo and Sang (2024) proposed a simpler idea: approximate the value function with a small set of basis functions (like polynomials), and optimize the basis coefficients alongside $\theta$. There is no neural network involved. The whole thing is one optimization call.

## Key Equations

$$
(\hat\theta, \hat\alpha) = \arg\max_{\theta, \alpha} \; \ell^f(\theta) - \frac{\lambda}{2} \|\alpha\|^2,
$$

where $V(s;\alpha) = \Psi(s)^\top \alpha$ is a polynomial or Fourier approximation and the penalty keeps the basis from overfitting. The action values under this approximation are

$$
Q(s,a;\theta,\alpha) = r_\theta(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, \Psi(s')^\top \alpha.
$$

## Pseudocode

```
SEES(D, features, p, beta, sigma, basis_type, M, lambda):
  1. Pick a basis: polynomials, Fourier terms, etc., with M terms
  2. Approximate V(s) = sum of basis functions weighted by alpha
  3. Compute Q from the approximated V
  4. Jointly optimize theta and alpha to maximize penalized likelihood
  5. Standard errors via Schur complement (marginalizes out alpha)
  6. Return theta, SEs
```

## Strengths and Limitations

SEES scales to state spaces over 100,000 because the cost is $O(M)$ regardless of how many states there are. The entire estimation is a single L-BFGS-B optimization call with no training loops, making it fast and straightforward to run.

The limitation is projection bias. If the value function has sharp discontinuities or mathematically rough boundaries, the sieve basis approximates them poorly. Manual basis tuning (choosing the right type and number of terms) is required. For problems where the expected value curve is smooth, SEES is the fastest path to structural estimates at scale.

SEES is the right choice for large state spaces when deep learning is overkill and you want valid standard errors from a single optimization call.

## References

- Luo, Y. & Sang, Y. (2024). Sieve-Based Estimation of Economic Structural Models.
