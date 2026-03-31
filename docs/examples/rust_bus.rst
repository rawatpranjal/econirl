Rust Bus Engine Replacement
===========================

This example replicates Rust (1987) on the bus engine replacement dataset. A fleet manager decides each month whether to keep running a bus engine or replace it. The operating cost rises with mileage and the replacement cost is fixed.

The state space has 90 mileage bins and 2 actions. All four structural estimators recover the same operating cost and replacement cost parameters.

.. code-block:: python

   from econirl import NFXP, CCP, NNES, TDCCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()

   nfxp  = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp   = CCP(discount=0.9999, num_policy_iterations=5).fit(df, state="mileage_bin", action="replaced", id="bus_id")

   print(nfxp.params_)  # {'theta_c': 0.001, 'RC': 3.07}
   print(ccp.params_)   # {'theta_c': 0.001, 'RC': 3.07}

.. list-table:: Parameter Recovery
   :header-rows: 1

   * - Estimator
     - theta_c
     - RC
     - Time
   * - NFXP
     - 0.0010
     - 3.072
     - 172s
   * - CCP (K=5)
     - 0.0010
     - 3.072
     - 0.1s
   * - NNES
     - 0.0012
     - 2.927
     - 4s
   * - TD-CCP
     - 0.0011
     - 3.076
     - 2s

NFXP and CCP recover identical parameters. CCP is 1700 times faster because it avoids the inner Bellman loop. The neural estimators (NNES and TD-CCP) get close but introduce small approximation error from the value network. This is expected. Neural methods are designed for high-dimensional problems where exact methods cannot run.

Counterfactual analysis
-----------------------

NFXP supports counterfactual policy simulation. The question is what happens to replacement behavior if the replacement cost doubles.

.. code-block:: python

   cf = nfxp.counterfactual(RC=6.0)
   print(cf.policy[50, :])  # P(keep|mileage=50), P(replace|mileage=50)

With the higher replacement cost, the manager keeps engines running longer. The probability of replacement at mileage 50 drops from 12 percent to 3 percent.

.. image:: /_static/rust_bus_counterfactual.png
   :alt: Counterfactual analysis showing replacement probability and value function under different replacement costs.
   :width: 100%

The left panel shows how the replacement probability shifts across mileage levels when the replacement cost changes. Halving the cost makes the manager replace earlier. Doubling it makes the manager delay replacement until higher mileage. The right panel shows the corresponding value function. A lower replacement cost raises the value at every mileage level because the manager has a cheaper exit option.
