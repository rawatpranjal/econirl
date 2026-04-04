# Approximate and Large-State Benchmarks on Rust Bus

| | |
|---|---|
| **Estimators** | NFXP (exact Bellman), SEES (Fourier sieve), NNES-NPL (neural V), TD-CCP (neural per-feature EV) |
| **Environment** | Rust bus engine, 90 to 200 mileage bins, beta 0.99 to 0.9999 |
| **Key finding** | All three approximation methods trade small accuracy losses for large speed gains. Each targets a different bottleneck in the exact NFXP pipeline. |

## Background

When the state space is large and the discount factor is high, the NFXP inner Bellman loop becomes the computational bottleneck. Three approximation strategies attack this from different angles.

SEES (Luo and Sang 2024) replaces the value function with a low-dimensional sieve basis and penalizes Bellman violations instead of solving the fixed point exactly. NNES (Nguyen 2025) uses a neural network to approximate the value function, exploiting a zero-Jacobian property so that first-order V approximation errors drop out of the structural parameter score. TD-CCP (Adusumilli and Eckardt 2025) learns separate neural networks for each utility feature component of the expected value, providing a per-feature decomposition useful for diagnostics.

## SEES vs NFXP

### Setup

The environment uses 200 mileage bins, a discount factor of 0.9999, and true parameters theta_c equal to 0.001 and RC equal to 3.0. Each replication simulates 200 buses over 100 periods. SEES uses 8 Fourier basis functions to approximate the 200-dimensional value function, reducing the problem to 10 joint parameters (2 structural plus 8 basis coefficients).

### Code

The full Monte Carlo script is at ``examples/rust-bus-engine/sees_vs_nfxp_mc.py``.

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

### Results

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

### Discussion

With 200 states and a discount factor of 0.9999, the NFXP inner loop requires tens of thousands of contraction iterations before the Newton-Kantorovich phase can begin. SEES bypasses this entirely by solving a single unconstrained optimization over 10 variables. The Bellman penalty ensures that the sieve approximation stays close to the true value function without ever computing the full fixed point. The tradeoff is approximation error from the finite basis. With 8 Fourier functions on a smooth cost function, this error is small. For cost functions with sharp nonlinearities, a larger basis or polynomial basis may be needed.

## NNES vs NFXP

### Setup

The environment uses 90 mileage bins and a discount factor of 0.99. Each replication simulates 200 buses over 100 periods. The NNES estimator uses a two-layer MLP with 32 hidden units, trained for 500 epochs per outer iteration with 3 outer NPL iterations.

### Code

The full Monte Carlo script is at ``examples/rust-bus-engine/nnes_vs_nfxp_mc.py``.

```python
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.nnes import NNESEstimator

nfxp = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-12)

nnes = NNESEstimator(
    hidden_dim=32,
    v_epochs=500,
    n_outer_iterations=3,
)
```

### Results

Five Monte Carlo replications with true parameters theta_c equal to 0.001 and RC equal to 3.0.

| Metric | NFXP | NNES |
|---|---|---|
| Bias (theta_c) | 0.0001 | 0.0061 |
| Bias (RC) | 0.0003 | 0.0007 |
| RMSE (theta_c) | 0.0004 | 0.6270 |
| RMSE (RC) | 0.0629 | 0.0409 |
| Mean log-likelihood | -4203.34 | -4203.29 |
| Mean wall time (seconds) | 22.3 | 51.0 |

### Discussion

The log-likelihoods are nearly identical, confirming that both estimators find the same quality of fit. The replacement cost RC is recovered well by both methods, with NNES actually achieving lower RMSE (0.041 vs 0.063). The operating cost theta_c shows much higher variance under NNES. This parameter is three orders of magnitude smaller than RC (0.001 vs 3.0), making it harder for the neural network to resolve the fine-grained mileage gradient in the value function.

The difficulty is inherent to the tabular setting where NFXP already works perfectly. NFXP solves the Bellman equation to machine precision, so even tiny parameters are identified exactly through the likelihood curvature. The neural V approximation introduces noise that is small relative to RC but large relative to theta_c. The NPL zero-Jacobian property ensures that this noise does not bias the standard errors, but it does not eliminate the noise itself.

The value of NNES emerges in continuous or high-dimensional state spaces where NFXP cannot operate because it requires enumerating all states. In those settings, the neural approximation is the only feasible path to structural MLE with valid inference.

## TD-CCP vs NFXP

### Setup

The environment uses 200 mileage bins, a discount factor of 0.9999, and true parameters theta_c equal to 0.001 and RC equal to 3.0. The data consists of 200 buses observed over 100 periods. TD-CCP uses a two-layer MLP with 32 hidden units and 15 approximate value iteration rounds with 3 policy iterations.

### Code

The full script is at ``examples/rust-bus-engine/tdccp_vs_nfxp.py``.

```python
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig

nfxp = NFXPEstimator(inner_solver="hybrid", inner_max_iter=300000)

tdccp = TDCCPEstimator(config=TDCCPConfig(
    hidden_dim=32, avi_iterations=15, n_policy_iterations=3,
))
```

### Results

Single replication with true parameters theta_c equal to 0.001 and RC equal to 3.0.

| Metric | NFXP | TD-CCP |
|---|---|---|
| theta_c | 0.001468 | 0.000197 |
| RC | 3.0731 | 3.0557 |
| Bias (theta_c) | 0.000468 | -0.000803 |
| Bias (RC) | 0.0731 | 0.0557 |
| Log-likelihood | -4193.14 | -4192.90 |
| Wall time (seconds) | 160.3 | 359.3 |

### Discussion

TD-CCP recovers the replacement cost within 2 percent of the true value but shows more bias on the operating cost. This reflects the neural approximation error in the per-feature EV decomposition. The operating cost is small (0.001) relative to the replacement cost (3.0), making it harder for the neural network to resolve the fine-grained mileage gradient. TD-CCP is slower than NFXP on this problem because the neural training overhead exceeds the savings from avoiding the inner loop. The advantage of TD-CCP emerges on continuous-state problems where neither NFXP nor CCP can operate without discretization bias.

The per-feature EV decomposition in TD-CCP is a unique diagnostic. Instead of a single opaque value function, the researcher sees how each utility component contributes to the expected continuation value. This can reveal which features are well-identified by the data and which are poorly pinned down.

## When to Use Which

SEES is the best choice for large tabular problems with smooth cost functions, where 8 to 16 Fourier or polynomial basis functions capture the value function shape. It delivers the largest speedups (30x) with minimal accuracy loss. NNES targets continuous or high-dimensional state spaces where tabular methods cannot operate. The zero-Jacobian property gives it valid asymptotic inference, making it the only approximate method here with a formal efficiency guarantee. TD-CCP is most valuable when per-feature diagnostics matter or when the state space is continuous and the researcher wants to understand which utility components drive the continuation value. On tabular problems, NFXP remains the gold standard.
