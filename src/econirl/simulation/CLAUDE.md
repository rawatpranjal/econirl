# simulation/ - Data Generation and Counterfactual Analysis

## Synthetic Data (`synthetic.py`)

**simulate_panel(env, n_individuals, n_periods, seed)** -> Panel. Computes optimal policy from true parameters via value iteration, then simulates trajectories by sampling actions from the logit policy and transitions from the environment.

**simulate_panel_from_policy(problem, transitions, policy, initial_distribution, ...)** -> Panel. Simulates without an environment object -- useful for counterfactual simulations with estimated policies.

**run_monte_carlo(env, estimator, utility, n_replications, ...)** -> MonteCarloResult. Repeatedly simulates data and estimates parameters. Computes bias, RMSE, and 95% CI coverage. MonteCarloResult holds arrays of shape `(n_replications, n_params)`.

## Counterfactual Analysis (`counterfactual.py`)

**counterfactual_policy(result, new_parameters, utility, problem, transitions)** -> CounterfactualResult. Solves for new optimal policy under changed parameters and compares to baseline from estimation result.

**counterfactual_transitions(result, new_transitions, ...)** -> CounterfactualResult. Analyzes behavior change under different state dynamics.

**simulate_counterfactual(...)**: Generates synthetic data under both baseline and counterfactual policies for aggregate comparison.

**compute_stationary_distribution(policy, transitions)**: Power iteration for ergodic distribution.

**elasticity_analysis(result, utility, problem, transitions, parameter_name, pct_changes)**: Sensitivity of policy to parameter perturbations.

## Gotchas

- `simulate_panel` uses `env.step()` for transitions, so the environment's internal RNG state matters. Always pass `seed`.
- Monte Carlo results use numpy arrays, not torch tensors (for scipy.stats compatibility).
- `compute_stationary_distribution` uses 1000 iterations of power method -- may not converge for non-ergodic chains.
