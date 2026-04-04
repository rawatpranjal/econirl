# SEES vs NFXP on Large State Spaces

| | |
|---|---|
| **Estimators** | NFXP (SA-then-NK hybrid), SEES (Fourier basis, K=8) |
| **Environment** | Rust bus engine, 200 mileage bins, beta 0.9999 |
| **Key finding** | SEES is 30 times faster with comparable parameter recovery. |

## Background

NFXP nests a full Bellman solve inside each optimizer step. When the state space is large and the discount factor is high, the contraction mapping converges slowly and the inner loop becomes the computational bottleneck. SEES (Luo and Sang 2024) avoids the inner loop entirely by approximating V(s) with a small set of basis functions and penalizing Bellman equation violations. The value function is represented as V(s) equal to Psi(s) times alpha, where Psi is a Fourier basis of dimension K and alpha are basis coefficients optimized jointly with the structural parameters theta. The penalty omega times the squared Bellman violation pushes the sieve approximation toward the true value function.

## Setup

The data generating process uses a Rust bus engine with 200 mileage bins, a discount factor of 0.9999, and the standard operating cost and replacement cost parameters. Each replication simulates 200 buses over 100 periods. The large state space (500 bins vs the usual 90) makes the NFXP inner loop expensive because each contraction iteration operates on a 500-dimensional vector and the high discount factor requires many iterations for convergence. SEES uses only 8 Fourier basis functions to approximate the 200-dimensional value function, reducing the problem to 10 joint parameters (2 structural plus 8 basis coefficients).

## Code

The full Monte Carlo script is at ``examples/rust-bus-engine/sees_vs_nfxp_mc.py``. The core estimator setup is:

```python
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.sees import SEESEstimator

env = RustBusEnvironment(num_mileage_bins=200, discount_factor=0.9999)

nfxp = NFXPEstimator(inner_solver="hybrid", inner_max_iter=300000)

sees = SEESEstimator(
    basis_type="fourier",
    basis_dim=8,
    penalty_weight=10.0,
)
```

## Results

Five Monte Carlo replications with true parameters theta_c equal to 0.001 and RC equal to 3.0.

| Metric | NFXP | SEES |
|---|---|---|
| Mean bias (theta_c) | 0.000054 | -0.000691 |
| Mean bias (RC) | 0.0005 | -0.0006 |
| RMSE (theta_c) | 0.0003 | 0.0007 |
| RMSE (RC) | 0.0607 | 0.0492 |
| Mean log-likelihood | -4239.98 | -4239.39 |
| Mean wall time (seconds) | 120.4 | 4.0 |
| Speedup | 1x | 30x |

## Discussion

The key advantage of SEES is computational. With 200 states and a discount factor of 0.9999, the NFXP inner loop requires tens of thousands of contraction iterations before the Newton-Kantorovich phase can begin. SEES bypasses this entirely by solving a single unconstrained optimization over 10 variables (2 structural parameters plus 8 basis coefficients). The Bellman penalty ensures that the sieve approximation stays close to the true value function without ever computing the full fixed point. The tradeoff is that SEES introduces approximation error from the finite basis. With 8 Fourier functions on a smooth cost function, this error is small. For cost functions with sharp nonlinearities, a larger basis or polynomial basis may be needed.
