Rust Bus Engine Replacement
===========================

.. image:: /_static/mdp_schematic_rust_bus.png
   :alt: Rust bus MDP structure showing mileage states with keep and replace actions.
   :width: 80%
   :align: center

This example replicates Rust (1987) on the bus engine replacement dataset. A fleet manager decides each month whether to keep running a bus engine or replace it. The operating cost rises with mileage and the replacement cost is fixed. The state space has 90 mileage bins and 2 actions.

.. image:: /_static/rust_bus_counterfactual.png
   :alt: Counterfactual analysis showing replacement probability and relative value function under varying replacement cost and operating cost.
   :width: 100%

Quick start
-----------

The sklearn-style API fits any estimator in one call. All four estimators below take the same inputs and return parameters through the same ``params_`` and ``se_`` attributes.

.. code-block:: python

   from econirl import NFXP, CCP, NNES, SEES, TDCCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp  = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp   = CCP(discount=0.9999, num_policy_iterations=3).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   nnes  = NNES(discount=0.9999, v_epochs=300, n_outer_iterations=2).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   tdccp = TDCCP(discount=0.9999, avi_iterations=15, epochs_per_avi=20, n_policy_iterations=2).fit(df, state="mileage_bin", action="replaced", id="bus_id")

One panel and many estimators
------------------------------

The same ``TrajectoryPanel`` can also be passed directly to several estimators when you want one comparison pipeline. This is especially useful for neural IRL because the panel object stays in the common econirl format. ``TrajectoryPanel`` stores arrays in JAX. ``NeuralAIRL`` trains its networks in Torch. The estimator handles that boundary internally during minibatching, so user code does not need to convert arrays by hand.

.. code-block:: python

   import jax.numpy as jnp

   from econirl import CCP, MCEIRL, NFXP, NeuralAIRL, RewardSpec, TrajectoryPanel
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()
   panel = TrajectoryPanel.from_dataframe(
       df,
       state="mileage_bin",
       action="replaced",
       id="bus_id",
   )

   features = jnp.zeros((90, 2, 2), dtype=jnp.float32)
   features = features.at[:, 0, 0].set(-jnp.arange(90, dtype=jnp.float32))
   features = features.at[:, 1, 1].set(-1.0)
   reward_spec = RewardSpec(features, names=["theta_c", "RC"])

   nfxp = NFXP(n_states=90, discount=0.9999).fit(panel, reward=reward_spec)
   ccp = CCP(n_states=90, discount=0.9999).fit(panel, reward=reward_spec)
   mce = MCEIRL(n_states=90, n_actions=2, discount=0.9999).fit(
       panel,
       reward=reward_spec,
   )
   airl = NeuralAIRL(n_actions=2, discount=0.9999, max_epochs=50, patience=10).fit(
       panel,
       features=reward_spec,
   )

The structural estimators use the panel directly. ``NeuralAIRL`` uses the same panel too and only converts a minibatch when the neural training step starts. That lets one Rust bus example compare classic DDC estimators and neural IRL estimators without changing the data container.

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
   * - SEES (Fourier, K=12)
     - -0.0006
     - 2.997
     - NaN
     - NaN
     - -1902
     - 8s

NFXP, CCP, and MCE-IRL recover identical parameters. CCP avoids the inner Bellman loop entirely and matches NFXP in a fraction of the time. MCE-IRL reaches the same answer from the IRL side, confirming the theoretical equivalence between maximum causal entropy IRL and logit DDC estimation. The neural estimators (NNES and TD-CCP) get close but introduce small approximation error from the value network. This is expected. Neural methods are designed for high-dimensional problems where exact methods cannot run.

NeuralGLADIUS needs tuning on this problem. We ran a quick sweep on simulated linear Rust bus data with 250 individuals and 80 periods because the true policy is known in that setting. The lower learning rate performs best among the tested settings, but every GLADIUS configuration remains far from the NFXP benchmark.

.. list-table:: NeuralGLADIUS Hyperparameter Sweep (250 individuals, 80 periods, simulated)
   :header-rows: 1

   * - Estimator
     - Config
     - Epochs
     - Policy MAE
     - Policy Corr
     - Param MAE
     - Time
   * - NFXP
     - Reference
     - -
     - 0.0053
     - -
     - -
     - -
   * - NeuralGLADIUS
     - Smaller lr 1e-4
     - 309
     - 0.4528
     - 0.9928
     - 1.5069
     - 96.3s
   * - NeuralGLADIUS
     - Bigger batch 2048
     - 230
     - 0.4877
     - 0.8188
     - 1.5297
     - 37.9s
   * - NeuralGLADIUS
     - Tikhonov annealing
     - 500
     - 0.4936
     - 0.9584
     - 1.5121
     - 194.2s
   * - NeuralGLADIUS
     - Baseline 64x2
     - 171
     - 0.4948
     - 0.8432
     - 1.5340
     - 52.2s
   * - NeuralGLADIUS
     - Longer training 2000 epochs
     - 402
     - 0.5036
     - 0.9170
     - 1.4983
     - 149.1s
   * - NeuralGLADIUS
     - High Bellman weight 10
     - 207
     - 0.5052
     - 0.8429
     - 1.5357
     - 60.4s
   * - NeuralGLADIUS
     - Joint updates
     - 201
     - 0.5065
     - 0.9401
     - 1.5323
     - 112.8s
   * - NeuralGLADIUS
     - Bigger net 128x3
     - 142
     - 0.5074
     - 0.9436
     - 1.5192
     - 82.0s

This sweep suggests that the main lever is optimization rather than model size. Reducing the learning rate improves policy recovery more than deeper networks, longer training, or stronger Bellman penalties. The full sweep is reproducible with ``python3 scripts/gladius_hyperparam_sweep.py``.

SEES (Luo and Sang 2024) approximates the value function with a Fourier sieve basis of dimension 12 and jointly optimizes structural parameters and basis coefficients via penalized MLE. It recovers RC within 0.003 of the true value. The operating cost parameter is close to zero, reflecting the difficulty of identifying a small parameter through basis function approximation. Standard errors are unavailable because the Schur complement Hessian is singular at this solution. On the 90-state bus problem the computational advantage over NFXP is modest (8 seconds versus 0.1 seconds), but the same approach scales to state spaces where NFXP inner loop is infeasible.

.. code-block:: python

   from econirl import SEES
   sees = SEES(discount=0.9999, basis_type="fourier", basis_dim=12, penalty_lambda=0.001)
   sees.fit(df, state="mileage_bin", action="replaced", id="bus_id")

MCE-IRL inference pipeline
--------------------------

The MCE-IRL estimator can also be used through its own sklearn-style wrapper with full inference output. The features are linear and quadratic functions of normalized mileage.

.. code-block:: python

   from econirl.estimation import MCEIRLEstimator as MCEIRL

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
