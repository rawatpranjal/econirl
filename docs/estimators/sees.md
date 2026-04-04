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

## Diagnostics and Guarantees

Identification requires the same conditions as NFXP. The utility features must vary across actions, and the transition matrix must be known. The additional requirement is that the sieve basis must be rich enough to approximate the true value function. If the value function has sharp discontinuities or features that the chosen basis cannot capture, the resulting projection bias contaminates the structural parameter estimates. Smooth value functions are well-approximated by both Fourier and polynomial bases at moderate dimensions.

Convergence is determined by a single L-BFGS-B optimization call over the joint parameter vector of structural parameters theta and basis coefficients alpha. The optimizer terminates when the gradient norm falls below 1e-6 or after 500 iterations. Because there is no inner loop, there is no distinction between inner and outer convergence. The L2 penalty on the basis coefficients (default lambda of 0.01) regularizes the optimization and prevents the sieve from overfitting.

Standard errors are computed via the Schur complement of the numerical Hessian. The full Hessian is computed over the joint parameter space of theta and alpha, then the alpha block is marginalized out to yield the correct marginal Hessian for theta. This approach gives valid standard errors for the structural parameters when the sieve dimension is fixed. At the sieve boundary, where the approximation error is comparable to the statistical error, the standard errors may understate the true uncertainty because they do not account for the projection bias.

The main limitation is manual basis selection. The user must choose the basis type (Fourier or polynomial) and the number of terms. Too few terms produce projection bias, while too many can overfit. In practice, 8 to 16 Fourier terms work well for smooth value functions on problems with up to a few hundred states. The L2 penalty helps control overfitting but cannot eliminate basis misspecification.

The default configuration uses a Fourier basis with 8 terms, an L2 penalty of 0.01, L-BFGS-B with a gradient tolerance of 1e-6, and a maximum of 500 iterations. The basis is constructed over normalized state values in the range of 0 to 1. These defaults are designed for problems where the value function is reasonably smooth.
