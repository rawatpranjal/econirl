# SEES

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Luo and Sang (2024) | Linear | Yes | Schur complement | Yes |

## What this estimator does

NFXP and CCP both require computation that scales with the full state space, making them impractical when the state space exceeds a few thousand. Luo and Sang (2024) propose SEES, which approximates the value function with a small set of deterministic basis functions (Fourier or polynomial) and imposes the Bellman equation as a penalty rather than a constraint. The computational cost is $O(K)$ where $K$ is the number of basis functions, independent of the state space size. No inner loop is solved and no neural network is trained.

The sieve approximation replaces $V(s)$ with the linear expansion $V(s;\alpha) = \Psi(s)^\top \alpha$, where $\Psi(s)$ is a basis vector. As the penalty weight $\omega$ grows and $K = |\mathcal{S}|$, SEES reduces to MPEC. By using $K \ll |\mathcal{S}|$, SEES solves a much smaller optimization problem. The trade-off is projection bias when the value function has sharp features that the chosen basis cannot represent.

The estimator is asymptotically efficient. The standard errors are computed via the Schur complement of the joint Hessian, marginalizing out the basis coefficients to give the correct Fisher information for the structural parameters.

## How it works

Estimation reduces to a single L-BFGS-B call that jointly maximizes the penalized criterion

$$
\max_{\theta, \alpha} \; \sum_{i=1}^N \log \pi(a_i \mid s_i; \theta, \alpha) - \omega \sum_{s} (V(s;\alpha) - T_\sigma(V;\theta)(s))^2.
$$

The continuation values $\mathbb{E}[\Psi(s') | s, a] = P_a \Psi$ are precomputed once as a matrix multiply for each action, so no inner loop is needed during optimization. Standard errors use the Schur complement of the joint $(\theta, \alpha)$ Hessian to produce the marginal Fisher information for $\theta$ alone.

## When to use it

SEES is the right choice when the state space is too large for NFXP or CCP but the value function is smooth enough to be well-approximated by a small basis. On the Rust bus benchmark, SEES achieves the same log-likelihood as NFXP while running an order of magnitude faster. The advantage grows with state space size. If the value function has sharp discontinuities, increase $K$ until parameter estimates stabilize. For settings where even a smooth basis is insufficient, NNES uses a neural network as a universal approximator at the cost of a more complex alternating optimization.

## References

- Luo, Y. and Sang, Y. (2024). Sieve-Based Estimation of Structural Models. Working paper.

The full derivation, algorithm, and simulation results are in the [SEES primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/sees.pdf).
