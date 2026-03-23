# core/ - Types, Bellman Operators, Solvers

## Types (`types.py`)

- **DDCProblem**: Frozen dataclass. Fields: `num_states`, `num_actions`, `discount_factor` (default 0.9999), `scale_parameter` (default 1.0). Validates ranges in `__post_init__`.
- **Trajectory**: Mutable dataclass. Fields: `states`, `actions`, `next_states` (all `torch.Tensor` of shape `(T,)`), optional `individual_id`, `metadata`. Supports `.to(device)`.
- **Panel**: List of Trajectory objects. Key properties: `num_observations` (total across all trajectories), `num_individuals`. Methods: `get_all_states()`, `get_all_actions()`, `get_all_next_states()` return concatenated tensors. `Panel.from_numpy()` classmethod groups by individual_id.

## Bellman Operator (`bellman.py`)

**SoftBellmanOperator** expects transitions shape `(num_actions, num_states, num_states)` where `transitions[a, s, s'] = P(s'|s,a)`. This is validated in `__post_init__`.

Key equations:
```
Q(s,a) = u(s,a) + beta * sum_s' P(s'|s,a) V(s')
V(s) = sigma * log(sum_a exp(Q(s,a) / sigma))
```

`BellmanResult` is a NamedTuple with fields: `Q` (S,A), `V` (S,), `policy` (S,A).

Helper: `compute_flow_utility(params, feature_matrix)` computes `einsum("sak,k->sa", ...)`.

## Solvers (`solvers.py`)

All return `SolverResult` with: `Q`, `V`, `policy`, `converged`, `num_iterations`, `final_error`.

- **value_iteration**: Simple contraction. Linear convergence with modulus beta.
- **policy_iteration**: Alternates evaluation (matrix solve or iterative) and improvement. Faster for high beta.
- **hybrid_iteration**: Contraction until `switch_tol`, then Newton-Kantorovich for quadratic convergence. Best for high beta (10-100x faster than pure VI).
- **solve()**: Convenience dispatcher accepting `method="value"|"policy"|"hybrid"`.

## Gotchas

- Transition shape is `(A, S, S)` everywhere in core. Some estimation code uses `(S, A, S)` -- always verify before passing.
- High discount factors (beta > 0.99) make VI slow. Use `hybrid_iteration` or reduce beta for testing.
- `SolverResult.converged=False` does not raise -- always check this field.
