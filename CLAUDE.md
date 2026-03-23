# CLAUDE.md - Project Context for Claude Code

## Project Overview

**econirl** is a Python library for structural estimation and inverse reinforcement learning (IRL) in dynamic discrete choice models. It bridges econometrics (NFXP, CCP) with machine learning (MaxEnt IRL, MCE IRL, GAIL, AIRL).

## Key Architecture

```
src/econirl/
├── core/           # Types (DDCProblem, Panel, Trajectory), Bellman operators, solvers
├── estimation/     # 10 estimators: NFXP, CCP, MaxEnt IRL, MCE IRL, Max Margin,
│   │                 GCL, TD-CCP, GLADIUS, GAIL, AIRL
│   └── adversarial/  # GAIL and AIRL implementations
├── environments/   # RustBus, MultiComponentBus, Gridworld
├── preferences/    # LinearUtility, ActionDependentReward, NeuralCost
├── inference/      # Standard errors, identification diagnostics, EstimationSummary
├── simulation/     # Synthetic data generation, Monte Carlo, counterfactuals
└── visualization/  # Policy and value function plots
```

Each subdirectory has its own CLAUDE.md with detailed interface documentation. See:
- `src/econirl/core/CLAUDE.md` - Types, Bellman operator, solvers
- `src/econirl/estimation/CLAUDE.md` - All 10 estimators and base contract
- `src/econirl/environments/CLAUDE.md` - Environment base class and 3 implementations
- `src/econirl/preferences/CLAUDE.md` - Utility function protocol and LinearUtility
- `src/econirl/inference/CLAUDE.md` - SE methods, identification, EstimationSummary
- `src/econirl/simulation/CLAUDE.md` - Data generation, Monte Carlo, counterfactuals
- `tests/CLAUDE.md` - Test conventions, fixtures, running tests

## Critical Implementation Details

### MCE IRL Expected Features (IMPORTANT)
The `_compute_expected_features()` method in `mce_irl.py` MUST iterate over the **empirical state sequence** from demonstrations, NOT use the stationary distribution:

```python
# CORRECT: Iterate over empirical states
for traj in panel.trajectories:
    for t in range(len(traj)):
        s = traj.states[t].item()
        for a in range(n_actions):
            feature_sum += policy[s, a] * feature_matrix[s, a, :]

# WRONG: Using stationary distribution (causes parameter recovery failure)
# return torch.einsum("s,sa,sak->k", state_visitation, policy, feature_matrix)
```

### Parameter Identification in IRL
- Rewards are only identified up to constants (Kim et al. 2021, Cao & Cohen 2021)
- For well-conditioned optimization, normalize parameters to unit norm
- Normalize features to [-1, 1] range
- Rust bus parameters (0.001, 3.0) have poor scaling - consider normalization

### Transition Matrix Conventions
- Estimators expect: `(n_actions, n_states, n_states)` i.e., `transitions[a, s, s']`
- Some internal code uses: `(n_states, n_actions, n_states)` i.e., `transitions[s, a, s']`
- Always check the expected format when passing transitions

## Testing Commands

```bash
# Run quick tests (skip slow)
python3 -m pytest tests/ -v -m "not slow"

# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_mce_irl_action_features.py -v

# Run with coverage
python3 -m pytest tests/ --cov=econirl
```

## Common Debugging

### MCE IRL not converging
1. Check feature normalization (should be [-1, 1])
2. Check parameter initialization (don't use zeros)
3. Verify inner loop (soft VI) converges - increase `inner_max_iter` for high gamma
4. Try smaller learning rate if oscillating

### NFXP slow convergence
1. High discount factor (gamma > 0.99) requires many inner iterations
2. Consider lowering gamma for testing
3. Check that transitions are properly normalized (rows sum to 1)

## Key References

- Rust (1987): Optimal Replacement of GMC Bus Engines
- Ziebart (2010): Maximum Causal Entropy IRL
- Kim et al. (2021): Reward Identification in IRL
- Cao & Cohen (2021): Identifiability in IRL
