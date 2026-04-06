# NNES

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Nguyen (2025) | Linear | Yes | Valid (orthogonality) | Yes |

## What this estimator does

Sieve methods like SEES approximate the value function with deterministic basis functions, but the number of terms needed grows exponentially in the state dimension. Neural networks are universal approximators that avoid this curse of dimensionality, but plugging a neural network into a structural estimation framework creates a problem. The approximation error in the value function contaminates the parameter estimates and invalidates the standard errors.

Nguyen (2025) solves this by embedding the neural value function inside the Nested Pseudo-Likelihood (NPL) framework. The critical insight is the zero Jacobian property. At the true conditional choice probabilities, the derivative of the NPL policy-iteration operator with respect to the value function vanishes. First-order approximation errors in the neural V-network drop out of the likelihood score, producing Neyman orthogonality. The estimator is $\sqrt{n}$-consistent and achieves the semiparametric efficiency bound without bias correction.

The information matrix is block-diagonal between $\theta$ and the V-network parameters. This means standard errors from the pseudo-likelihood Hessian are valid without the Schur complement marginalization that SEES requires. Only the NPL variant has this property. The NFXP Bellman variant, which trains $V$ to minimize the Bellman residual directly, lacks Neyman orthogonality and produces unreliable standard errors.

## How it works

The estimator alternates between two phases. Phase 1 trains the V-network via supervised regression on the NPL target, which is the continuation value implied by the current CCPs through the Hotz-Miller inversion. An auxiliary Bellman penalty encourages the network to satisfy the Bellman equation. Phase 2 maximizes the pseudo-log-likelihood

$$
\hat\theta = \arg\max_\theta \sum_{i=1}^N \log \pi(a_i \mid x_i; \theta, V_\gamma)
$$

over $\theta$ alone, treating the V-network as fixed. CCPs are updated from the new parameter estimate, and the outer loop repeats. One iteration suffices for $\sqrt{n}$-consistency, but finite-sample performance improves with additional iterations. Standard errors come from the Hessian of the pseudo-log-likelihood at the final $\hat\theta$.

## When to use it

NNES is the right choice for continuous or high-dimensional state spaces where NFXP and CCP cannot scale and the value function is too complex for a sieve basis. It inherits the statistical elegance of NPL while using neural networks to overcome the curse of dimensionality. The main practical limitation relative to NFXP is sensitivity to initialization and the number of outer iterations. For settings where transition densities are also unavailable, TD-CCP achieves similar properties without requiring them.

## References

- Nguyen, H. (2025). Neural Network Estimation of Structural Models. Working paper.

The full derivation, algorithm, and simulation results are in the [NNES primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/nnes.pdf).
