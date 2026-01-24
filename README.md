# econirl

The StatsModels of IRL - A Python package bridging Structural Econometrics and Inverse Reinforcement Learning.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from econirl import RustBusEnvironment, LinearUtility, NFXPEstimator
from econirl.simulation import simulate_panel

# Create environment with known parameters
env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)

# Simulate data
panel = simulate_panel(env, n_individuals=500, n_periods=100)

# Estimate
utility = LinearUtility.from_environment(env)
estimator = NFXPEstimator()
result = estimator.estimate(panel, utility, env.problem_spec, env.transition_matrices)

# View results
print(result.summary())
```

## Features

- Economist-friendly API (utility, preferences, characteristics)
- StatsModels-style `summary()` output
- Multiple estimation methods (NFXP, with CCP/MaxEnt planned)
- Rich inference (standard errors, confidence intervals, hypothesis tests)
- Gymnasium-compatible environments
- Counterfactual analysis and visualization

## Replication Packages

### Rust (1987) - Bus Engine Replacement

Replicate the classic dynamic discrete choice paper:

```python
from econirl.replication.rust1987 import table_v_structural

# Reproduce Table V structural estimates
table = table_v_structural(groups=[4])
print(table)
```

See the [full replication notebook](examples/rust_1987_replication.ipynb) for details.

## License

MIT
