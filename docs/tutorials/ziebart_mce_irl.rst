Ziebart MCE IRL Replication
===========================

These experiments replicate the gridworld results from Ziebart (2008, 2010) using econirl. MCE IRL (Maximum Causal Entropy) accounts for causal structure in the state visitation computation, while MaxEnt IRL (Maximum Entropy) does not. The comparison shows where the causal entropy formulation improves policy recovery.

Replication on 5x5 Gridworld
-----------------------------

The replication script recovers reward parameters on a five-by-five gridworld (25 states, 5 actions) with three features (step penalty, terminal reward, distance weight) from 100 expert trajectories of 30 periods each (3,000 observations). The discount factor is 0.95.

IRL rewards are identified only up to an additive constant and multiplicative scale (Kim et al. 2021), so the normalized direction matters more than raw parameter values.

.. code-block:: text

   MCE IRL (Ziebart 2010):
     Converged: True
     Log-likelihood: -3965.10
     Cosine similarity: 0.68
     Policy accuracy: 100.0%
     KL(true || model): 0.001497

   MaxEnt IRL (Ziebart 2008):
     Converged: False
     Log-likelihood: -3972.61
     Policy accuracy: 88.0%
     KL(true || model): 0.008353

MCE IRL achieves perfect policy accuracy at all 25 states, with KL divergence from the true policy five times lower than MaxEnt IRL. The causal formulation matters because the non-causal state visitation computation in MaxEnt IRL propagates errors through the forward pass, producing systematically different choice probabilities.

.. code-block:: text

   Policy at key states:
   State 0 (corner):
     Action       True      MCE   MaxEnt
     Right      0.4997   0.5000   0.4887
     Down       0.4997   0.5000   0.4887

   State 23 (adjacent to goal):
     Action       True      MCE   MaxEnt
     Right      1.0000   1.0000   0.9934

MCE IRL matches the true policy exactly. MaxEnt IRL spreads probability mass onto suboptimal actions because its state visitation frequencies are computed without the causal direction constraint.

The full script is at ``examples/ziebart-mce-irl/ziebart_mce_irl_replication.py``.

Extended Benchmark
------------------

The benchmark extends the comparison across three reward specifications on the same 5x5 gridworld with 2,000 trajectories of 50 periods, adding NFXP as a structural baseline. A 70/30 train/test split evaluates in-sample, out-of-sample, and transfer (stochastic transitions) performance.

**Case 1: State-action rewards (full identification).** Features vary by action. MCE IRL and NFXP both achieve cosine similarity 0.9999 to the true reward vector (RMSE 0.015). MaxEnt IRL recovers the wrong direction (cosine similarity negative 0.72).

.. code-block:: text

   Reward Recovery:
   Param                  True      MCE IRL   MaxEnt IRL         NFXP
   move_cost           -0.5000      -0.4885      -0.3742      -0.4885
   goal_approach        2.0000       2.0241      -2.1949       2.0241
   northward            0.1000       0.0855       1.0418       0.0855
   eastward             0.1000       0.1052       1.0471       0.1051
   Cosine sim                        0.9999      -0.7234       0.9999

   Transfer performance (LL/obs):
     MCE IRL:    -1.3578
     MaxEnt IRL: -1.9023
     NFXP:       -1.3578

MaxEnt IRL fails catastrophically on transfer because the non-causal state visitation produces a reward vector pointing in a qualitatively different direction.

**Case 2: Rust-style (action-dependent feature mapping).** Two state features mapped differently per action (like mileage cost vs replacement cost). All three methods achieve high cosine similarity (MCE and NFXP at 0.9999, MaxEnt at 0.9990). The simpler reward structure reduces the gap between causal and non-causal formulations.

.. code-block:: text

   Reward Recovery:
   Param                  True      MCE IRL   MaxEnt IRL         NFXP
   operating_cost       2.0000       2.0265       1.4396       2.0265
   move_cost            0.3000       0.3288       0.2817       0.3288
   Cosine sim                        0.9999       0.9990       0.9999

**Case 3: Pure state-only (weak identification).** Same features for all actions. Policy driven by transition-mediated value differences. MCE IRL and NFXP are near-perfect (cosine 1.0000, RMSE 0.013). MaxEnt IRL inflates the magnitude (RMSE 1.54) but preserves the direction (cosine 0.9999).

.. code-block:: text

   Reward Recovery:
   Param                  True      MCE IRL   MaxEnt IRL         NFXP
   goal_dist            3.0000       2.9821       5.1568       2.9821
   center_dist          0.5000       0.5050       0.7937       0.5050
   Cosine sim                        1.0000       0.9999       1.0000

All three cases use deterministic gridworld transitions. MaxEnt IRL can recover rewards under deterministic dynamics when the feature structure is simple. In Cases 2 and 3, where the reward uses only two parameters, MaxEnt achieves cosine similarity above 0.999 and matches MCE IRL and NFXP on in-sample and out-of-sample policy performance. The failure in Case 1 is not about transition stochasticity but about feature complexity. Four action-dependent parameters (including directional preferences) create enough structure for the non-causal state visitation computation to diverge from the causal one, even though the transitions themselves are deterministic.

The practical takeaway is that MaxEnt IRL is a viable approximation in deterministic environments with simple reward specifications, but MCE IRL is strictly safer. When features vary across actions and the reward structure is rich, the causal formulation prevents the silent sign-flip errors visible in Case 1.

The full script is at ``examples/ziebart-mce-irl/run_gridworld.py``.
