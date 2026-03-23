# tests/ - Test Suite

## Running Tests

```bash
# Quick (skip slow parameter recovery and scaling tests)
python -m pytest tests/ -v -m "not slow"

# All tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_mce_irl_core.py -v

# With coverage
python -m pytest tests/ --cov=econirl
```

## Conventions

- Framework: pytest with `-v --tb=short` recommended.
- Slow tests: Marked with `@pytest.mark.slow` (parameter recovery, large-scale, neural estimators). Found in: `test_gladius.py`, `test_td_ccp.py`, `test_nfxp_convergence.py`, integration tests.
- Fixtures: Defined in `conftest.py`. Key fixtures:
  - `rust_env` / `rust_env_small`: Standard and small Rust bus environments.
  - `small_panel` / `medium_panel` / `large_panel`: Pre-simulated panel data.
  - `utility` / `utility_small`: LinearUtility from environments.
  - `transitions` / `transitions_small`: Transition matrices.
  - `bellman_operator`, `optimal_policy`, `optimal_value`: Pre-computed solutions.
  - `assert_valid_policy`, `assert_valid_value_function`: Validation helpers.
  - `simple_problem`, `synthetic_panel`: MCE IRL specific fixtures.

## Directory Structure

- `tests/`: Unit tests for individual modules (one test file per source module).
- `tests/integration/`: Cross-module integration tests (estimator pipelines, dataset loading, Rust replication).
- `tests/benchmarks/`: Cross-estimator comparison benchmarks (`bench_ccp_nfxp.py`).

## Test Naming

- Unit tests: `test_{module_name}.py` (e.g., `test_nfxp_sklearn.py`, `test_gridworld.py`).
- Core/sklearn tests: `test_{estimator}_core.py` and `test_{estimator}_sklearn.py` for estimators with both interfaces.
- Integration: `test_{feature}_estimation.py` or `test_{feature}_estimators.py`.

## Gotchas

- MCE IRL tests use `mce_irl_seed` fixture for numpy random state isolation.
- Estimation tolerance fixture is 0.5 (loose) -- parameter recovery is approximate.
- `rust_env_small` uses `discount_factor=0.99` (not 0.9999) for faster convergence in tests.
