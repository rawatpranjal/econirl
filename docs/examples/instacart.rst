Instacart Grocery Reorder
=========================

.. image:: /_static/instacart_overview.png
   :alt: Grocery reorder DDC showing state dependence in reorder rates by purchase frequency, recency effect on reorder probability, and state visitation heatmap across purchase and recency buckets.
   :width: 100%

This example applies structural estimation to a grocery reorder problem modeled on the Instacart Market Basket Analysis data. A consumer decides each week whether to reorder a product category from a delivery platform. The state captures purchase history (how many times the consumer has ordered before) and recency (how long since the last order). The action is skip or reorder.

The environment has 20 states arranged as 5 purchase frequency buckets by 4 recency buckets. Reordering increases the purchase frequency bucket and resets recency to zero. Skipping increases recency and lets the purchase frequency bucket decay slowly. This captures two stylized facts from real grocery data: consumers who have ordered more in the past are more likely to reorder (state dependence), and consumers who ordered recently are more likely to reorder again soon (recency effect).

Quick start
-----------

.. code-block:: python

   from econirl.environments.instacart import InstacartEnvironment
   from econirl.datasets.instacart import load_instacart

   env = InstacartEnvironment(discount_factor=0.95)
   panel = load_instacart(n_individuals=1000, n_periods=52, as_panel=True)

Estimation
----------

The three utility parameters capture habit formation, recency sensitivity, and the fixed cost of placing an order.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.preferences.linear import LinearUtility
   import jax.numpy as jnp

   utility = LinearUtility(features=env.feature_matrix, parameter_names=env.parameter_names)
   init = jnp.zeros(3)

   nfxp = NFXPEstimator(env.problem_spec, env.transition_matrices)
   result = nfxp.fit(panel, utility, init_params=init)

   for name, val in zip(env.parameter_names, result.parameters):
       print(f"{name}: {float(val):.4f}")

The true parameters are habit strength of 0.3, recency effect of 0.5, and reorder cost of negative 0.2. A positive habit strength means consumers who have reordered before are more likely to reorder again. A positive recency effect means recent purchasers are more likely to reorder. The negative reorder cost represents the hassle of placing an order.

Marketing interpretation
------------------------

This model connects to the Erdem and Keane (1996) tradition of structural brand choice models. The habit strength parameter measures state dependence, which marketing researchers use to separate true brand loyalty from spurious correlation driven by unobserved heterogeneity. The recency effect captures stockpiling and inventory management behavior. The reorder cost reflects switching costs and search frictions on the platform.

Counterfactual experiments are straightforward. Reducing the reorder cost (for example through a promotional free delivery offer) shifts the reorder rate upward. The structural model predicts not just the immediate effect but the long-run equilibrium through the transition dynamics, since more reorders today increase future purchase frequency which further increases future reorder rates.
