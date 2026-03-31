econirl
=======

.. code-block:: bash

   pip install econirl

econirl recovers structural parameters from sequential choice data. Every estimator shares one interface and returns standard errors.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rust_bus

   df = load_rust_bus()
   nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp  = CCP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   print(nfxp.params_)  # {'theta_c': 0.001, 'RC': 3.07}
   print(ccp.params_)   # {'theta_c': 0.001, 'RC': 3.07}  — same answer

Estimators
----------

econirl has two families of estimator. Linear reward estimators learn exact structural parameters from a reward function of the form R(s,a) = theta times phi(s,a). Neural reward estimators learn a nonlinear reward function via neural networks and project onto features for approximate theta.

.. list-table::
   :header-rows: 1

   * -
     - Linear Reward (exact theta)
     - Neural Reward (projected theta)
   * - Structural MLE
     - ``NFXP`` (Rust 1987), ``NNES`` (Nguyen 2025)
     - ``NeuralGLADIUS`` (Kang et al. 2025)
   * - Reduced-form DDC
     - ``CCP`` (Hotz and Miller 1993), ``TDCCP``
     - ``NeuralAIRL`` (Fu et al. 2018)
   * - Maximum Entropy IRL
     - MCE-IRL (Ziebart 2010, coming soon)
     -

Linear reward estimators approximate the value function with neural networks while keeping the reward linear in parameters. The structural parameters are exact and the standard errors are valid. Neural reward estimators approximate the reward function itself. The structural parameters are extracted by least-squares projection, and the projection R-squared tells you how linear the learned reward actually is. The formal equivalence between maximum entropy IRL and logit DDC was established by Zeng et al. (2025) and Geng et al. (2017).

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   examples/index
   api/index
   references
