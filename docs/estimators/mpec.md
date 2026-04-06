# MPEC

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Structural | Su and Judd (2012) | Linear | Yes | Analytical (MLE) | No |

## What this estimator does

NFXP nests the Bellman solve inside the optimizer. MPEC takes a different approach. Instead of solving the Bellman equation at each candidate parameter vector, it treats the value function $V$ as explicit decision variables and enforces the Bellman equation $V = T(V;\theta)$ as a constraint. This eliminates the inner fixed-point loop entirely and replaces it with a single constrained optimization over $(\theta, V)$ jointly.

Su and Judd (2012) showed that this formulation can be solved efficiently by standard nonlinear programming solvers. On the Rust bus with $\beta = 0.9999$, MPEC with KNITRO solved in 0.5 seconds versus 188 seconds for NFXP with successive approximation. However, Iskhakov, Rust and Schjerning (2016) later showed that when NFXP uses Newton-Kantorovich instead of successive approximation, the speed advantage of MPEC largely disappears at moderate state spaces. MPEC remains relevant for very large state spaces where forming the NK Jacobian becomes expensive.

At convergence, MPEC and NFXP solve the same MLE problem. Identification requires the same rank condition on the feature matrix as NFXP. Standard errors are computed by the same implicit function theorem formula because the Bellman constraint is exactly satisfied at the optimum.

## How it works

The estimator solves the augmented Lagrangian formulation

$$
\min_{\theta, V} \; -\mathcal{L}(\theta, V) + \lambda^\top [V - T(V;\theta)] + \frac{\rho}{2} \|V - T(V;\theta)\|^2,
$$

where $\mathcal{L}$ is the log-likelihood, $\lambda$ are Lagrange multipliers, and $\rho$ is a penalty weight that grows across outer iterations until the Bellman constraint is satisfied. Each inner subproblem is solved by L-BFGS-B. A warm start from value iteration at the initial $\theta_0$ is critical for giving the optimizer a feasible starting point. Standard errors use the same formula as NFXP because both reach the same MLE fixed point.

## When to use it

MPEC is an alternative to NFXP for settings where the inner Bellman loop is the bottleneck. In practice, NFXP-NK is faster at moderate state spaces below a few hundred states. MPEC's advantage emerges at larger state spaces where forming the NK Jacobian becomes expensive. The augmented Lagrangian penalty schedule requires some tuning and the Bellman constraint violation should be checked at convergence.

## References

- Su, C.-L. and Judd, K. L. (2012). Constrained Optimization Approaches to Estimation of Structural Models. *Econometrica*, 80(5), 2213-2230.
- Iskhakov, F., Rust, J., and Schjerning, B. (2016). Comment on "Constrained Optimization Approaches to Estimation of Structural Models." *Econometrica*, 84(1), 365-370.

The full derivation, algorithm, and simulation results are in the [MPEC primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/mpec.pdf).
