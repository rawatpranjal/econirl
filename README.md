# econirl

The StatsModels of IRL - A Python package bridging Structural Econometrics and Inverse Reinforcement Learning.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from econirl import NFXP
from econirl.datasets import load_rust_bus

# Load data
df = load_rust_bus()

# Fit model
est = NFXP(n_states=90, discount=0.9999)
est.fit(df, state="mileage_bin", action="replaced", id="bus_id")

# View results
print(est.params_)   # {'theta_c': 0.00107, 'RC': 9.35}
print(est.summary())

# Simulate
sim = est.simulate(n_agents=100, n_periods=50)

# Counterfactual: what if RC doubled?
cf = est.counterfactual(RC=est.params_["RC"] * 2)
```

### Available Estimators

| Estimator | Description |
|-----------|-------------|
| `NFXP` | Nested Fixed Point (Rust 1987) |
| `CCP` | Conditional Choice Probability (Hotz-Miller) |

All estimators share the same interface:
- `est.fit(df, state=, action=, id=)` - fit model
- `est.params_` - parameter estimates
- `est.se_` - standard errors
- `est.summary()` - formatted results
- `est.simulate()` - simulate choices
- `est.counterfactual()` - policy analysis

## Features

- Economist-friendly API (utility, preferences, characteristics)
- StatsModels-style `summary()` output
- Multiple estimation methods (NFXP, CCP, with MaxEnt planned)
- Rich inference (standard errors, confidence intervals, hypothesis tests)
- Gymnasium-compatible environments
- Counterfactual analysis and visualization

## Available Datasets

econirl includes datasets for both DDC (structural econometrics) and IRL (inverse reinforcement learning).

### DDC Datasets

| Dataset | Domain | States | Actions | Complexity |
|---------|--------|--------|---------|------------|
| `load_rust_bus()` | Bus Repair | 1 (mileage) | 2 | Low |
| `load_keane_wolpin()` | Career | 3+ (exp, educ) | 4 | High |
| `load_robinson_crusoe()` | Production | 1 (inventory) | 2-3 | Low |

```python
from econirl.datasets import load_rust_bus, load_keane_wolpin, load_robinson_crusoe

# Rust (1987) - Bus engine replacement
df = load_rust_bus(original=True)
panel = load_rust_bus(as_panel=True, group=4)

# Keane & Wolpin (1994) - Career decisions
df_kw = load_keane_wolpin()
print(df_kw['choice'].value_counts())

# Robinson Crusoe - Pedagogical model
df_rc = load_robinson_crusoe(n_individuals=100, include_hunt=True)
```

### IRL Trajectory Datasets

| Dataset | Domain | Format | Best For |
|---------|--------|--------|----------|
| `load_tdrive()` | Taxi navigation | GPS trajectories | MaxEnt IRL, route preferences |
| `load_geolife()` | Human mobility | GPS + mode labels | Mobility IRL, activity patterns |
| `load_stanford_drone()` | Pedestrians/cyclists | Pixel trajectories | Social navigation IRL |
| `load_eth_ucy()` | Pedestrians | World coordinates | Benchmark trajectory prediction |

```python
from econirl.datasets import load_tdrive, load_geolife, load_stanford_drone, load_eth_ucy

# T-Drive: Taxi route data for MaxEnt IRL
trajectories = load_tdrive(as_trajectories=True, discretize=True, grid_size=100)

# Stanford Drone: Pedestrian paths
df = load_stanford_drone(scene="gates", agent_type="Pedestrian")

# ETH/UCY: Classic pedestrian benchmark
df = load_eth_ucy(scene="eth")
trajectories = load_eth_ucy(as_trajectories=True, discretize=True)
```

### Preprocessing Utilities

```python
from econirl.preprocessing import discretize_state, check_panel_structure

# Discretize continuous states
df['state_bin'] = discretize_state(df['mileage'], method='uniform', n_bins=90)

# Validate panel structure
result = check_panel_structure(df, id_col='id', period_col='period')
print(f"Valid: {result.valid}, Balanced: {result.is_balanced}")
```

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
