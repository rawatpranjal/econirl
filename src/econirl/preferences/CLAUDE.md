# preferences/ - Utility Function Specifications

## Protocol (`base.py`)

**UtilityFunction** protocol requires:
- Properties: `num_parameters`, `parameter_names`, `num_states`, `num_actions`
- `compute(parameters) -> Tensor(S, A)`: Flow utility matrix
- `compute_gradient(parameters) -> Tensor(S, A, K)`: Gradient w.r.t. parameters

**BaseUtilityFunction** ABC provides: `validate_parameters()`, `get_initial_parameters()` (defaults to zeros), `get_parameter_bounds()` (defaults to unbounded), optional `anchor_action` for identification normalization.

## LinearUtility (`linear.py`)

The primary utility specification: `U(s,a;theta) = theta . phi(s,a)`.

- Constructor takes `feature_matrix` of shape `(num_states, num_actions, num_features)`.
- `compute(params)` uses `einsum("sak,k->sa", feature_matrix, params)`.
- `compute_gradient(params)` returns the feature matrix (constant for linear utility).
- `compute_hessian(params)` returns zeros (linear in parameters).
- `from_environment(env)` classmethod extracts features and parameter names from a DDCEnvironment.
- `anchor_action` normalization subtracts anchor features from all actions for identification.

## Other Modules

- `action_reward.py`, `action_utility.py`: Action-dependent reward/utility specifications.
- `neural_cost.py`: Neural network cost function for deep IRL methods.
- `reward.py`: Reward function wrapper.

## Gotchas

- `feature_matrix` shape is `(S, A, K)` throughout the codebase. Do not confuse with `(S, K)`.
- For IRL, rewards are identified only up to additive constants. Use `anchor_action` or normalize to unit norm.
- `compute_gradient` takes `parameters` for interface consistency but LinearUtility ignores it (gradient is constant).
