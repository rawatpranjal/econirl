# environments/ - DDC Environments

## Base Class (`base.py`)

**DDCEnvironment** extends `gym.Env` (Gymnasium). Abstract properties that subclasses must implement:

- `num_states`, `num_actions`: State/action space sizes
- `transition_matrices`: Shape `(num_actions, num_states, num_states)` where `result[a, s, s'] = P(s'|s,a)`
- `feature_matrix`: Shape `(num_states, num_actions, num_features)`
- `true_parameters`: Dict mapping parameter names to values
- `parameter_names`: Ordered list of parameter name strings

Abstract methods: `_get_initial_state_distribution()`, `_compute_flow_utility(state, action)`, `_sample_next_state(state, action)`.

Convenience: `problem_spec` returns a `DDCProblem`, `compute_utility_matrix(params)` computes `einsum("sak,k->sa", ...)`, `get_true_parameter_vector()` returns tensor in canonical order.

## Environments

### RustBusEnvironment (`rust_bus.py`)
Rust (1987) bus engine replacement. 90 mileage bins (default), 2 actions (keep/replace), 2 features. Parameters: `operating_cost` (default 0.001), `replacement_cost` (default 3.0). Mileage transitions: stochastic +{0,1,2}; replace resets to 0.

### MultiComponentBusEnvironment (`multi_component_bus.py`)
K independent components, M mileage bins each. State space: M^K states via mixed-radix encoding. 2 actions, 3 features: `[replacement_cost, operating_cost, quadratic_cost]`. Raises `ValueError` for K >= 4 (state space too large for dense tensors).

### GridworldEnvironment (`gridworld.py`)
N x N grid, 5 actions (Left/Right/Up/Down/Stay). N^2 states indexed as `row * N + col`. Absorbing terminal at (N-1, N-1). 3 features: `[step_penalty, terminal_reward, distance_weight]`. Deterministic transitions.

### ShapeshifterEnvironment (`shapeshifter.py`)

Shape-shifting synthetic DGP used by the JSS deep-run Tier 4 cells to verify estimator-vs-paper alignment. Construct from a `ShapeshifterConfig` parameterized along eight axes: `reward_type` (linear or neural), `feature_type` (linear or neural), `action_dependent`, `stochastic_transitions`, `stochastic_rewards`, `num_periods` (None for infinite, int for finite), `discount_factor`, and `state_dim` (1 for scalar, >1 for product space). Total state count is `num_states ** state_dim`, capped at 4096. Neural reward and neural features are produced by frozen tanh MLPs initialized deterministically from `seed`. Ground truth is exact in every regime: linear infinite uses `hybrid_iteration`, finite-horizon uses `backward_induction`, neural-reward feeds the precomputed reward matrix to the same solvers. Per-cell expected estimator support is declared in `experiments/jss_deep_run/matrix.py`.

## Gotchas

- Transition convention is always `(num_actions, num_states, num_states)` -- this is enforced by `SoftBellmanOperator.__post_init__`.
- DDC environments are infinite horizon (`step()` never returns `terminated=True`).
- `feature_matrix` shape is `(S, A, K)`, not `(S, K)` -- features are state-action specific.
