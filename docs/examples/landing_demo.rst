Landing Demo: Four Estimators, One API
======================================

This is the minimal example from the landing page. It fits four different estimators to the same Rust bus dataset using the unified sklearn-style interface and prints a comparison table with parameters and standard errors.

.. code-block:: python

   from econirl import NFXP, CCP, NNES, TDCCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp  = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp   = CCP(discount=0.9999, num_policy_iterations=3).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   nnes  = NNES(discount=0.9999, v_epochs=300, n_outer_iterations=2).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   tdccp = TDCCP(discount=0.9999, avi_iterations=15, epochs_per_avi=20, n_policy_iterations=2).fit(df, state="mileage_bin", action="replaced", id="bus_id")

All four estimators take the same inputs and return parameters through the same ``params_`` and ``se_`` attributes. The structural estimators (NFXP and CCP) recover nearly identical parameters. The neural approximations (NNES and TD-CCP) get close but introduce small approximation error from the value network, which is expected since neural methods are designed for high-dimensional problems where exact methods cannot run.

.. list-table:: Parameter Comparison (original Rust bus data)
   :header-rows: 1

   * - Estimator
     - theta_c
     - RC
     - SE(theta_c)
     - SE(RC)
   * - NFXP
     - 0.0012
     - 3.07
     - 0.0003
     - 0.24
   * - CCP
     - 0.0012
     - 3.07
     - 0.0003
     - 0.24
   * - NNES
     - 0.03
     - 3.07
     - 0.01
     - 0.30
   * - TD-CCP
     - 0.001
     - 2.94
     - 0.001
     - 0.30

.. code-block:: bash

   python examples/landing_demo.py

This script runs in under 30 seconds and demonstrates the core value proposition: same problem, four algorithms, one API, all with standard errors.
