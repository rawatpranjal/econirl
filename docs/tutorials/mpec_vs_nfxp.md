# MPEC vs NFXP Parameter Recovery

| | |
|---|---|
| **Estimators** | NFXP (SA-then-NK hybrid), MPEC (SLSQP) |
| **Environment** | Rust bus engine, 90 mileage bins, beta 0.9999 |
| **Key finding** | Both recover identical MLE. MPEC is 12 times faster. |

## Background

NFXP (Rust 1987) nests a full Bellman solve inside each optimizer step. When the discount factor is high, contraction mapping converges slowly and the inner loop dominates computation time. MPEC (Su and Judd 2012) treats the value function as decision variables alongside the structural parameters. The Bellman equation becomes an equality constraint handled natively by the optimizer. Both methods produce the same maximum likelihood estimates because they solve the same mathematical problem through different numerical paths.

## Setup

The data generating process uses the standard Rust bus engine with 90 discretized mileage bins, two structural parameters (operating cost theta_c and replacement cost RC), and a discount factor of 0.9999. Each replication simulates 200 buses over 100 periods, yielding 20,000 observations. The NFXP estimator uses the hybrid inner solver that starts with successive approximation and switches to Newton-Kantorovich iterations once the error drops below 0.001. The MPEC estimator uses scipy SLSQP with an analytical constraint Jacobian computed via JAX.

## Code

The full Monte Carlo script is at ``examples/rust-bus-engine/mpec_vs_nfxp_mc.py``. The core estimator setup is:

```python
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mpec import MPECEstimator, MPECConfig

nfxp = NFXPEstimator(
    inner_solver="hybrid",
    inner_tol=1e-12,
    inner_max_iter=100000,
    switch_tol=1e-3,
)

mpec = MPECEstimator(
    config=MPECConfig(solver="slsqp", max_iter=500, constraint_tol=1e-8),
)
```

## Results

Five Monte Carlo replications with true parameters theta_c equal to 0.001 and RC equal to 3.0.

| Metric | NFXP | MPEC |
|---|---|---|
| Mean bias (theta_c) | 0.000054 | 0.000054 |
| Mean bias (RC) | 0.0005 | 0.0005 |
| RMSE (theta_c) | 0.000348 | 0.000348 |
| RMSE (RC) | 0.0607 | 0.0607 |
| Mean log-likelihood | -4239.98 | -4239.98 |
| Mean wall time (seconds) | 90.8 | 7.6 |
| Converged | 5 of 5 | 5 of 5 |

## Discussion

The parameter estimates and log-likelihoods are identical to machine precision, confirming that MPEC and NFXP solve the same optimization problem. The speed difference comes entirely from avoiding the inner fixed-point loop. At beta equal to 0.9999, the contraction mapping requires thousands of iterations before the Newton-Kantorovich phase can begin. MPEC sidesteps this by treating V as free variables and letting the SLSQP optimizer enforce the Bellman constraint directly. The speedup of roughly 12 times is consistent with Su and Judd (2012), who report similar gains on the same model. MPEC is the preferred method when the state space is moderate (hundreds of states) and the discount factor is high. For very large state spaces where the decision variable count grows with the number of states, SEES or NNES may be more appropriate.
