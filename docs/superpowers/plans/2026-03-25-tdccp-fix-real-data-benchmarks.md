# TD-CCP Fix + Real-Data Multi-Estimator Benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix TD-CCP's policy extraction bug and validate 9 focus estimators across 5 real-world datasets.

**Architecture:** Fix TD-CCP by adding a proper Bellman re-solve after parameter estimation. Then extend existing NGSIM and Beijing taxi examples to include more estimators, and create new Foursquare and Citibike examples with dataset-appropriate estimator selections.

**Tech Stack:** PyTorch, scipy, econirl estimators, pandas for data loading.

---

## Focus Estimators (9 total)

| # | Estimator | Type | Key Strength |
|---|-----------|------|-------------|
| 1 | NFXP-NK | Structural MLE | Gold standard, exact Bellman |
| 2 | CCP | Structural CCP | Fast, Hotz-Miller inversion |
| 3 | MCE-IRL | Entropy IRL | Bridge estimator, feature matching |
| 4 | TD-CCP | Neural CCP | Scales to large state spaces |
| 5 | NNES | Neural V + MLE | Neural value approx + structural |
| 6 | SEES | Sieve basis V | Basis function approximation |
| 7 | AIRL | Adversarial IRL | Reward recovery, dynamics-robust |
| 8 | f-IRL | Distribution matching | State-marginal matching |
| 9 | BC | Baseline | Supervised learning baseline |

## Dataset → Estimator Mapping

| Dataset | States | Actions | Estimators | Rationale |
|---------|--------|---------|------------|-----------|
| **Rust bus** | 90 | 2 | NFXP, CCP, MCE-IRL | Already done (gold standard) |
| **NGSIM** | 50 | 3 | NFXP, MCE-IRL, Deep MaxEnt | Medium scale, interpretable |
| **Beijing taxi** | 225-400 | 5 | MCE-IRL, TD-CCP | Spatial grid, tests neural VI |
| **Foursquare** | 40 | 10 | AIRL, Deep MaxEnt, BC | Sequential choice, 10 categories |
| **Citibike** | ~500 | ~50 | AIRL, GLADIUS, TD-CCP | High-dim station choice |

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/econirl/estimation/td_ccp.py` | Modify (lines 612-615) | Add Bellman re-solve after param estimation |
| `tests/test_td_ccp_bellman_fix.py` | Create | Verify TD-CCP policy matches NFXP |
| `examples/ngsim-lane-change/run_ngsim_irl.py` | Modify | Add Deep MaxEnt estimator |
| `examples/beijing-taxi/run_estimation.py` | Modify | Add TD-CCP estimator |
| `examples/foursquare-venue/run_estimation.py` | Create | AIRL + Deep MaxEnt + BC on venue choice |
| `src/econirl/datasets/citibike.py` | Create | Citibike trip data loader |
| `examples/citibike-station/run_estimation.py` | Create | AIRL + GLADIUS + TD-CCP on station choice |

---

### Task 1: Fix TD-CCP Bellman Re-solve

**Files:**
- Modify: `src/econirl/estimation/td_ccp.py:612-615`
- Create: `tests/test_td_ccp_bellman_fix.py`

**Root cause:** After estimating parameters via partial MLE, TD-CCP computes the final policy using EV networks trained under the data CCPs, not under the estimated parameters. The Bellman equation is violated — the policy doesn't match its own value function.

**Fix:** Replace lines 612-615 with a proper Bellman solve using `SoftBellmanOperator` + `hybrid_iteration` on the estimated reward parameters.

- [ ] **Step 1: Write failing test**

```python
"""Test that TD-CCP policy matches NFXP policy after Bellman re-solve."""
import torch
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.estimation.nfxp import NFXPEstimator


def test_td_ccp_policy_close_to_nfxp():
    """TD-CCP policy should be close to NFXP policy (max diff < 0.1)."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0,
        num_mileage_bins=90, discount_factor=0.9999,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)

    # NFXP (ground truth policy)
    nfxp = NFXPEstimator()
    nfxp_result = nfxp.estimate(panel=panel, utility=utility,
                                 problem=problem, transitions=transitions)

    # TD-CCP
    config = TDCCPConfig(
        avi_iterations=20, epochs_per_avi=30,
        n_policy_iterations=3, verbose=False,
    )
    tdccp = TDCCPEstimator(config=config)
    tdccp_result = tdccp.estimate(panel=panel, utility=utility,
                                   problem=problem, transitions=transitions)

    # Policy should be close (pre-fix this fails with max_diff > 0.8)
    max_diff = (nfxp_result.policy - tdccp_result.policy).abs().max().item()
    assert max_diff < 0.1, f"TD-CCP policy too far from NFXP: max_diff={max_diff:.4f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_td_ccp_bellman_fix.py -v`
Expected: FAIL with max_diff > 0.8

- [ ] **Step 3: Implement the fix in td_ccp.py**

In `_optimize()`, replace lines 612-615:

```python
# OLD (broken): uses EV networks from data CCPs
# v = self._compute_choice_values_fn(params_opt)
# policy = torch.nn.functional.softmax(v / sigma, dim=1)
# V = sigma * torch.logsumexp(v / sigma, dim=1)

# NEW: proper Bellman re-solve with estimated parameters
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration

self._log("Re-solving Bellman equation with estimated parameters")
operator = SoftBellmanOperator(problem, transitions.double())
flow_utility = utility.compute(params_opt).double()
bellman_result = hybrid_iteration(
    operator, flow_utility, tol=1e-10, max_iter=5000
)
policy = bellman_result.policy.float()
V = bellman_result.V.float()
```

Add the imports at the top of the file if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_td_ccp_bellman_fix.py -v`
Expected: PASS with max_diff < 0.1

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `python3 -m pytest tests/ -v -m "not slow" -k "td_ccp or tdccp"`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/econirl/estimation/td_ccp.py tests/test_td_ccp_bellman_fix.py
git commit -m "fix: TD-CCP Bellman re-solve after parameter estimation

TD-CCP was computing final policy using EV networks trained under data
CCPs, violating the Bellman equation. Now re-solves the Bellman equation
with estimated parameters via hybrid_iteration, matching NFXP policy."
```

---

### Task 2: Add Deep MaxEnt to NGSIM Example

**Files:**
- Modify: `examples/ngsim-lane-change/run_ngsim_irl.py`

The NGSIM example already runs MaxEnt, MCE-IRL, NFXP, and AIRL. Add Deep MaxEnt IRL to compare neural vs linear reward on real driving data.

- [ ] **Step 1: Add Deep MaxEnt to `run_all_three()` (rename to `run_all()`)**

After the AIRL block (~line 476), add:

```python
# 5. Deep MaxEnt IRL (Wulfmeier 2016) — neural reward network
try:
    from econirl.estimation.deep_maxent_irl import DeepMaxEntIRLEstimator
    print("\n-- Deep MaxEnt IRL (Wulfmeier 2016) -- Neural Reward --")
    t0 = time.time()
    est = DeepMaxEntIRLEstimator(
        hidden_dims=[32, 32],
        lr=1e-3,
        max_epochs=300,
        verbose=True,
    )
    r = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s, Converged: {r.converged}")
    if r.log_likelihood is not None:
        print(f"  LL: {r.log_likelihood:.2f}")
    results["Deep MaxEnt"] = (r, elapsed)
except Exception as e:
    print(f"  Deep MaxEnt failed: {e}")
    import traceback; traceback.print_exc()
```

- [ ] **Step 2: Run and verify output**

Run: `cd examples/ngsim-lane-change && python run_ngsim_irl.py`
Expected: All 5 estimators produce results. Deep MaxEnt should show LL comparable to MCE-IRL.

- [ ] **Step 3: Commit**

```bash
git add examples/ngsim-lane-change/run_ngsim_irl.py
git commit -m "feat: add Deep MaxEnt IRL to NGSIM benchmark"
```

---

### Task 3: Add TD-CCP to Beijing Taxi Example

**Files:**
- Modify: `examples/beijing-taxi/run_estimation.py`

The Beijing taxi example currently runs MCE-IRL and NFXP on a 15x15 grid (225 states, 5 actions). Add TD-CCP to test neural approximate VI on real spatial data.

- [ ] **Step 1: Add TD-CCP estimation section**

After the NFXP section (~line 250), before the benchmark table, add a new section:

```python
# =================================================================
# 5b. TD-CCP (Neural Approximate VI)
# =================================================================
print(f"\n[5b/6] TD-CCP (Neural Approximate VI)")
print(f"{'=' * 50}")

from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig

tdccp_config = TDCCPConfig(
    avi_iterations=20,
    epochs_per_avi=50,
    n_policy_iterations=5,
    hidden_dim=64,
    learning_rate=1e-3,
    verbose=False,
)
tdccp = TDCCPEstimator(config=tdccp_config)
t0 = time.time()
tdccp_result = tdccp.estimate(
    panel=train_panel, utility=reward_fn, problem=problem, transitions=train_transitions,
)
tdccp_time = time.time() - t0

print(f"  Converged: {tdccp_result.converged}, Time: {tdccp_time:.1f}s")
print(f"  Parameters:")
for name, val in zip(feature_names, tdccp_result.parameters.tolist()):
    print(f"    {name:<20} {val:>10.4f}")

tdccp_ll_train, tdccp_policy = compute_ll(
    tdccp_result.parameters, reward_fn, problem, train_transitions, train_panel
)
tdccp_ll_test, _ = compute_ll(
    tdccp_result.parameters, reward_fn, problem, train_transitions, test_panel
)
```

- [ ] **Step 2: Update benchmark table to include TD-CCP**

Extend the results table to show MCE IRL, NFXP, and TD-CCP side-by-side.

- [ ] **Step 3: Run and verify**

Run: `cd examples/beijing-taxi && python run_estimation.py --n-taxis 100 --grid-size 15`
Expected: TD-CCP produces parameters and policy. With the Bellman fix from Task 1, policy should be reasonable.

- [ ] **Step 4: Commit**

```bash
git add examples/beijing-taxi/run_estimation.py
git commit -m "feat: add TD-CCP to Beijing taxi benchmark"
```

---

### Task 4: Create Foursquare Venue Choice Example

**Files:**
- Create: `examples/foursquare-venue/run_estimation.py`

Foursquare NYC: 40 states (10 venue categories x 4 time bins), 10 actions (next category). Run AIRL, Deep MaxEnt, and BC.

- [ ] **Step 1: Create the example script**

The script should:
1. Load Foursquare data via `load_foursquare(as_panel=True)`
2. Also load as DataFrame for building transitions and features
3. Estimate transitions from data
4. Build feature matrix: (40, 10, K) with category embeddings and time features
5. Run AIRL (adversarial, reward recovery)
6. Run Deep MaxEnt (neural reward)
7. Run BC (baseline)
8. Print comparison table with LL, accuracy, time

```python
"""
Foursquare NYC: Sequential Venue Choice with IRL
=================================================

Applies AIRL, Deep MaxEnt IRL, and Behavioral Cloning to the
Foursquare NYC dataset (Yang et al. 2015) modeling venue choice.

State: (venue_category, time_of_day) - 40 states
Action: next venue category - 10 actions
Data: ~227K check-ins from 1,084 NYC users
"""
```

Key implementation details:
- Feature matrix: one-hot for category transitions + time-of-day indicators
- Transitions estimated from data frequency counts
- AIRL config: `reward_type="tabular"`, `max_rounds=200`
- Deep MaxEnt: `hidden_dims=[32, 32]`, `max_epochs=300`
- BC: default config

- [ ] **Step 2: Run and verify output**

Run: `cd examples/foursquare-venue && python run_estimation.py`
Expected: All three produce results. Compare LL and accuracy.

- [ ] **Step 3: Commit**

```bash
git add examples/foursquare-venue/run_estimation.py
git commit -m "feat: Foursquare venue choice example with AIRL, Deep MaxEnt, BC"
```

---

### Task 5: Create Citibike Loader

**Files:**
- Create: `src/econirl/datasets/citibike.py`

Citibike NYC: 2M trips in Jan 2024. Design as discrete station-cluster choice.

- [ ] **Step 1: Write the loader**

Design decisions:
- Cluster ~1,700 stations into ~50 zones using k-means on (lat, lng)
- State: (origin_zone, hour_bin, day_type) where hour_bin in {0..5} (4-hour bins), day_type in {weekday, weekend}
- Action: destination zone (50 zones)
- State space: 50 zones x 6 hour_bins x 2 day_types = 600 states
- Action space: 50 zones

```python
"""
Citibike NYC Trip Data Loader
=============================

Loads Citibike trip data and converts to a discrete choice problem:
- State: (origin_zone, hour_bin, day_type)
- Action: destination_zone
- Zones: k-means clusters of stations by geographic location
"""
```

The loader function `load_citibike_panel()` should:
1. Read CSV from `data/raw/citibike/`
2. Parse station lat/lng, cluster into N zones via k-means
3. Discretize time into (hour_bin, day_type)
4. Build Panel of user trips as trajectories
5. Return dict with panel, transitions, feature_matrix, problem, metadata

- [ ] **Step 2: Test the loader**

```python
from econirl.datasets.citibike import load_citibike_panel
data = load_citibike_panel(n_zones=50)
print(f"States: {data['problem'].num_states}")
print(f"Actions: {data['problem'].num_actions}")
print(f"Observations: {data['panel'].num_observations}")
```

- [ ] **Step 3: Commit**

```bash
git add src/econirl/datasets/citibike.py
git commit -m "feat: Citibike NYC dataset loader with zone clustering"
```

---

### Task 6: Create Citibike Station Choice Example

**Files:**
- Create: `examples/citibike-station/run_estimation.py`

Run AIRL, GLADIUS, and TD-CCP on Citibike data. This tests high-dimensional scalability.

- [ ] **Step 1: Create the example script**

```python
"""
Citibike NYC: Station Choice with Scalable IRL
===============================================

Applies AIRL, GLADIUS, and TD-CCP to Citibike trip data,
testing scalability of IRL/DDC estimators on high-dimensional
real-world discrete choice.

State: (origin_zone, hour_bin, day_type) - ~600 states
Action: destination_zone - ~50 actions
Data: ~2M trips from January 2024
"""
```

The script should:
1. Load Citibike data via `load_citibike_panel(n_zones=50)`
2. Train/test split (70/30)
3. Run AIRL (tabular reward, 200 rounds)
4. Run GLADIUS (Q-net + EV-net, 300 epochs)
5. Run TD-CCP (neural AVI, 5 policy iterations)
6. Print comparison table

- [ ] **Step 2: Run and verify**

Run: `cd examples/citibike-station && python run_estimation.py`
Expected: All three produce results. Compare LL, accuracy, runtime.

- [ ] **Step 3: Commit**

```bash
git add examples/citibike-station/run_estimation.py
git commit -m "feat: Citibike station choice example with AIRL, GLADIUS, TD-CCP"
```

---

### Task 7: Final integration commit

- [ ] **Step 1: Run all examples end-to-end**

```bash
# TD-CCP fix verified
python3 -m pytest tests/test_td_ccp_bellman_fix.py -v

# Existing examples still work
cd examples/ngsim-lane-change && python run_ngsim_irl.py
cd examples/beijing-taxi && python run_estimation.py --n-taxis 50 --grid-size 10

# New examples
cd examples/foursquare-venue && python run_estimation.py
cd examples/citibike-station && python run_estimation.py
```

- [ ] **Step 2: Push all commits**

```bash
git push
```
