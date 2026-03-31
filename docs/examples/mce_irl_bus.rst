MCE-IRL on Bus Engine Data
==========================

This example applies Maximum Causal Entropy IRL directly to the original Rust (1987) bus engine dataset. Unlike the main Rust bus example which compares four structural estimators, this one focuses on MCE-IRL specifically and demonstrates the full inference pipeline: parameter estimation, standard errors, reward function recovery, choice probability prediction, and model fit comparison against empirical replacement rates.

The state space has 90 mileage bins and 2 actions (keep running or replace). The features are linear and quadratic functions of normalized mileage.

.. code-block:: python

   from econirl.datasets import load_rust_bus
   from econirl.estimators.mce_irl import MCEIRL

   df = load_rust_bus()
   model = MCEIRL(
       n_states=90, n_actions=2, discount=0.99,
       feature_matrix=features,
       feature_names=["linear_cost", "quadratic_cost"],
       se_method="hessian", verbose=True,
   )
   model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
   print(model.summary())

The sklearn-style API handles everything in one call. The summary includes parameter estimates, standard errors computed from the numerical Hessian, and the log-likelihood.

The script then prints the recovered reward function at selected mileage bins, showing how the cost of operating an engine increases with mileage. It also compares model-predicted replacement probabilities against empirical rates from the data, demonstrating that the MCE-IRL model captures the pattern of increasing replacement rates at high mileage.

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

.. code-block:: bash

   python examples/mce-irl/mce_irl_bus_demo.py

A second script in the same directory (``mce_irl_bus_example.py``) runs a similar analysis with additional diagnostic output.

Reference
---------

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University.
