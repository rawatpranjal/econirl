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

Estimators differ along two axes. The first is whether the reward function is linear in parameters or learned by a neural network. Linear reward estimators recover exact structural parameters with valid standard errors. Neural reward estimators learn a flexible reward function and extract approximate parameters by least-squares projection. The projection R-squared tells you how much of the neural reward is explained by your linear features.

The second axis is whether the reward depends only on the state or on both the state and the action. State-only rewards R(s) fit problems where the value comes from where you are, such as route choice on a road network. State-action rewards R(s,a) fit problems where different actions have different costs or benefits at the same state, such as engine replacement or hotel search.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Linear Reward (exact theta, valid SEs)
     - Neural Reward (projected theta, pseudo-SEs)
   * - **R(s,a)** state-action
     - ``NFXP`` (Rust 1987), ``CCP`` (Hotz-Miller 1993), ``NNES`` (Nguyen 2025), ``TDCCP``
     - ``NeuralGLADIUS`` (Kang et al. 2025), ``NeuralAIRL`` (Fu et al. 2018)
   * - **R(s)** state-only
     - MCE-IRL (Ziebart 2010, coming soon)
     -

All estimators in the R(s,a) row accept a feature tensor of shape (S, A, K) where each state-action pair has its own feature vector. MCE-IRL in the R(s) row accepts a feature tensor of shape (S, K) that is broadcast to all actions. The formal equivalence between maximum entropy IRL and logit DDC was established by Zeng et al. (2025) and Geng et al. (2017). See :doc:`references` for the full list of papers.

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   loading_data
   estimators
   examples/index
   api/index
   references
