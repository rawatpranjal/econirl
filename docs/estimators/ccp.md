# CCP

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Hotz and Miller (1993), Aguirregabiria and Mira (2002) | Linear | Yes | Hessian | No |

## What this estimator does

NFXP solves the Bellman equation to machine precision at every candidate parameter vector, costing $O(|\mathcal{S}|^2)$ per inner iteration. CCP eliminates the inner loop entirely. Hotz and Miller (1993) showed that under additive separability and IID extreme value shocks, the observed choice probabilities contain all the information needed to recover continuation values without solving any fixed point. The idea is to invert the mapping from value functions to choice probabilities, replacing thousands of Bellman iterations with a single matrix inversion of cost $O(|\mathcal{S}|^3)$.

Aguirregabiria and Mira (2002) showed that iterating the Hotz-Miller procedure in CCP space, a process called Nested Pseudo-Likelihood (NPL), converges to the maximum likelihood estimator. A single Hotz-Miller step gives a consistent but inefficient estimator. Five to ten NPL iterations recover full MLE efficiency. The zero Jacobian property ensures that at the NPL fixed point, the estimator is first-order insensitive to CCP estimation error.

Identification requires the same rank condition on the feature matrix as NFXP. The transition matrix must be known or consistently estimated from data. CCP is the only method in the package that extends naturally to dynamic games, where NFXP requires computing Nash equilibria in the inner loop.

## How it works

Given CCPs $\hat\pi(a|s)$ estimated from data, the continuation value is recovered by inverting

$$
\bar{v} = (I - \beta F_\pi)^{-1} \left[\sum_a \pi(a) \odot (u(a;\theta) + e(a))\right],
$$

where $e(a,s) = \gamma_{\text{Euler}} - \log \pi(a|s)$ is the emax correction and $F_\pi$ is the policy-weighted transition matrix. This gives choice-specific values that are linear in $\theta$, so pseudo-likelihood maximization reduces to a standard logit. Each NPL iteration updates the CCPs from the new parameter estimate and re-inverts. Standard errors come from the Hessian of the full Bellman-constrained log-likelihood at the converged parameters.

## When to use it

CCP is the method of choice when inner-loop cost is the bottleneck, which occurs at high discount factors or moderate state spaces of 100 to 500 states. CCP requires the full transition matrix, so for continuous state spaces where transition densities are hard to estimate, TD-CCP is the appropriate alternative. A known limitation is that CCP-NPL can produce unreliable counterfactual predictions at high discount factors because small parameter errors compound through the matrix inversion when $\beta$ is near 1.

## References

- Hotz, V. J. and Miller, R. A. (1993). Conditional Choice Probabilities and the Estimation of Dynamic Models. *The Review of Economic Studies*, 60(3), 497-529.
- Aguirregabiria, V. and Mira, P. (2002). Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models. *Econometrica*, 70(4), 1519-1543.

The full derivation, algorithm, and simulation results are in the [CCP primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/ccp.pdf).
