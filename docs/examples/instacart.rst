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
   panel = load_instacart(n_individuals=2000, n_periods=52, as_panel=True)

Estimation
----------

Three estimators recover the utility parameters from 1600 training consumers over 52 weekly periods (83200 observations).

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   nfxp = NFXPEstimator()
   result = nfxp.estimate(panel, utility, env.problem_spec, env.transition_matrices)

.. list-table:: Parameter Recovery (1600 consumers, 52 weeks)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP
     - MCE-IRL
   * - habit_strength
     - 0.3000
     - 0.3789
     - 0.2015
     - 0.3786
   * - recency_effect
     - 0.5000
     - 0.5279
     - 0.2500
     - 0.5279
   * - reorder_cost
     - -0.2000
     - -0.3048
     - 0.2203
     - -0.3045

NFXP and MCE-IRL produce nearly identical estimates and recover the correct sign and relative magnitude of all three parameters. The habit strength is positive (consumers who reorder more keep reordering), the recency effect is positive (recent purchasers reorder sooner), and the reorder cost is negative (placing an order has a hassle cost). NFXP standard errors are 0.17 for habit strength, 0.06 for recency effect, and 0.15 for reorder cost.

Counterfactual analysis
-----------------------

The free delivery promotion sets the reorder cost to zero, simulating a promotional offer that eliminates the hassle of placing an order. The welfare change is positive 4.33.

.. list-table:: Reorder Probability Under Free Delivery
   :header-rows: 1

   * - State
     - Baseline
     - Free Delivery
     - Change
   * - 0 orders, 0-3 days
     - 0.666
     - 0.732
     - +0.066
   * - 0 orders, 8-14 days
     - 0.642
     - 0.715
     - +0.073
   * - 3-5 orders, 0-3 days
     - 0.684
     - 0.747
     - +0.063
   * - 3-5 orders, 8-14 days
     - 0.662
     - 0.731
     - +0.069
   * - 11+ orders, 0-3 days
     - 0.683
     - 0.746
     - +0.063
   * - 11+ orders, 8-14 days
     - 0.661
     - 0.731
     - +0.069

The free delivery promotion increases reorder rates by 6 to 7 percentage points across all states. The lift is slightly larger for consumers with stale purchase history (8 to 14 days since last order) than for recent purchasers (0 to 3 days), suggesting that the promotion is most effective at reactivating lapsed customers.

.. list-table:: Reorder Cost Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -100% (free)
     - +4.329
     - 0.068
   * - -50%
     - +2.114
     - 0.035
   * - -25%
     - +1.044
     - 0.018
   * - +25%
     - -1.017
     - 0.019
   * - +50%
     - -2.006
     - 0.038
   * - +100%
     - -3.897
     - 0.077

The elasticity is approximately symmetric around the baseline. Each 25 percent reduction in reorder cost raises welfare by about 1.0 and shifts the reorder probability by about 1.8 percentage points. The response is nearly linear across the range tested.

Marketing interpretation
------------------------

This model connects to the Erdem and Keane (1996) tradition of structural brand choice models. The habit strength parameter measures state dependence, which marketing researchers use to separate true brand loyalty from spurious correlation driven by unobserved heterogeneity. The recency effect captures stockpiling and inventory management behavior. The reorder cost reflects switching costs and search frictions on the platform.

Counterfactual experiments are straightforward. Reducing the reorder cost (for example through a promotional free delivery offer) shifts the reorder rate upward. The structural model predicts not just the immediate effect but the long-run equilibrium through the transition dynamics, since more reorders today increase future purchase frequency which further increases future reorder rates.
