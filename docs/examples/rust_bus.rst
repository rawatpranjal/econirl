Rust Bus Engine Replacement
===========================

.. image:: /_static/rust_bus_counterfactual.png
   :alt: Counterfactual analysis showing replacement probability and relative value function under varying replacement cost and operating cost.
   :width: 100%

This example replicates Rust (1987) on the bus engine replacement dataset. A fleet manager decides each month whether to keep running a bus engine or replace it. The operating cost rises with mileage and the replacement cost is fixed. The state space has 90 mileage bins and 2 actions.

Quick start
-----------

The sklearn-style API fits any estimator in one call. All four estimators below take the same inputs and return parameters through the same ``params_`` and ``se_`` attributes.

.. code-block:: python

   from econirl import NFXP, CCP, NNES, TDCCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp  = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp   = CCP(discount=0.9999, num_policy_iterations=3).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   nnes  = NNES(discount=0.9999, v_epochs=300, n_outer_iterations=2).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   tdccp = TDCCP(discount=0.9999, avi_iterations=15, epochs_per_avi=20, n_policy_iterations=2).fit(df, state="mileage_bin", action="replaced", id="bus_id")

.. list-table:: Parameter Recovery (200 individuals, 100 periods, simulated)
   :header-rows: 1

   * - Estimator
     - theta_c
     - RC
     - SE(theta_c)
     - SE(RC)
     - LL
     - Time
   * - NFXP
     - 0.0012
     - 3.011
     - 0.0003
     - 0.24
     - -4263
     - 0.1s
   * - CCP (K=20)
     - 0.0012
     - 3.011
     - 0.0003
     - 0.24
     - -4263
     - 0.1s
   * - MCE-IRL
     - 0.0012
     - 3.011
     - 0.0003
     - 0.24
     - -4263
     - 0.1s
   * - NNES
     - 0.0315
     - 3.073
     - 0.01
     - 0.30
     - -4264
     - 16s
   * - TD-CCP
     - 0.0011
     - 2.943
     - 0.001
     - 0.30
     - -4265
     - 130s

NFXP, CCP, and MCE-IRL recover identical parameters. CCP avoids the inner Bellman loop entirely and matches NFXP in a fraction of the time. MCE-IRL reaches the same answer from the IRL side, confirming the theoretical equivalence between maximum causal entropy IRL and logit DDC estimation. The neural estimators (NNES and TD-CCP) get close but introduce small approximation error from the value network. This is expected. Neural methods are designed for high-dimensional problems where exact methods cannot run.

MCE-IRL inference pipeline
--------------------------

The MCE-IRL estimator can also be used through its own sklearn-style wrapper with full inference output. The features are linear and quadratic functions of normalized mileage.

.. code-block:: python

   from econirl.estimators.mce_irl import MCEIRL

   model = MCEIRL(
       n_states=90, n_actions=2, discount=0.99,
       feature_matrix=features,
       feature_names=["linear_cost", "quadratic_cost"],
       se_method="hessian", verbose=True,
   )
   model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
   print(model.summary())

The summary includes parameter estimates, standard errors computed from the numerical Hessian, and the log-likelihood. The script prints the recovered reward function at selected mileage bins, showing how the cost of operating an engine increases with mileage. It also compares model-predicted replacement probabilities against empirical rates.

.. list-table:: Model Fit at Selected Mileage Bins
   :header-rows: 1

   * - Mileage Bin
     - Empirical P(replace)
     - Model P(replace)
   * - 0
     - 0.00
     - 0.00
   * - 30
     - 0.01
     - 0.01
   * - 60
     - 0.04
     - 0.03
   * - 80
     - 0.10
     - 0.08

Counterfactual analysis
-----------------------

Structural estimation enables counterfactual policy simulation. Once we have estimated the operating cost and replacement cost parameters, we can ask what happens to replacement behavior under different economic conditions.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy

   new_params = result.parameters.clone()
   new_params[1] *= 2  # replacement_cost
   cf = counterfactual_policy(result, new_params, utility, problem, transitions)
   print(cf.policy[50, :])  # P(keep|mileage=50), P(replace|mileage=50)

The top row of the figure shows how replacement probability changes across mileage levels under two types of parameter variation. The top-left panel varies the replacement cost RC. Halving the replacement cost to 1.5 raises the replacement probability at every mileage level because the manager can afford to replace engines more freely. Tripling it to 9.0 suppresses replacement almost entirely, forcing the manager to run engines at high mileage. The top-right panel varies the per-mile operating cost. Doubling the operating cost makes high-mileage operation expensive, so the manager replaces earlier. Halving it makes running the engine cheap, so the manager holds off on replacement.

The bottom row shows the relative value function V(s) minus V(0) under each scenario. Plotting relative to mileage bin zero reveals the shape that is hidden when looking at the raw value function, where the absolute level with discount factor 0.9999 dominates the scale. The value function declines with mileage in every scenario because higher mileage means higher cumulative operating costs ahead. The decline is steepest when the replacement cost is high or the operating cost is high, because the manager faces larger losses from running a worn engine and has fewer attractive exit options.

Running the examples
--------------------

.. code-block:: bash

   # Quick start: 4 estimators, one API, 30 seconds
   python examples/landing_demo.py

   # Full NFXP vs MCE-IRL comparison on simulated and original data
   python examples/rust-bus-engine/compare_nfxp_mceirl.py

   # MCE-IRL inference pipeline with reward recovery and model fit
   python examples/mce-irl/mce_irl_bus_demo.py

Reference
---------

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University.
