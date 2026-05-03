Instacart Grocery Reorder
=========================

.. image:: /_static/mdp_schematic_instacart.png
   :alt: Instacart MDP structure showing frequency by recency state grid with reorder and skip actions.
   :width: 80%
   :align: center

This example applies structural estimation to a grocery reorder problem modeled on the Instacart Market Basket Analysis data. A consumer decides each week whether to reorder a product category from a delivery platform. The state captures purchase history (how many times the consumer has ordered before) and recency (how long since the last order). The action is skip or reorder.

.. image:: /_static/instacart_overview.png
   :alt: Grocery reorder DDC showing state dependence in reorder rates by purchase frequency, recency effect on reorder probability, and state visitation heatmap across purchase and recency buckets.
   :width: 100%

The environment has 20 states arranged as 5 purchase frequency buckets by 4 recency buckets. Reordering increases the purchase frequency bucket and resets recency to zero. Skipping increases recency and lets the purchase frequency bucket decay slowly. This captures two stylized facts from real grocery data: consumers who have ordered more in the past are more likely to reorder (state dependence), and consumers who ordered recently are more likely to reorder again soon (recency effect).

Quick start
-----------

.. code-block:: python

   from econirl.environments.instacart import InstacartEnvironment

   env = InstacartEnvironment(discount_factor=0.95)
   panel = env.generate_panel(n_individuals=2000, n_periods=52)

Estimation
----------

Three estimators recover the utility parameters from 1600 training consumers over 52 weekly periods (83,200 observations). The true parameters are a habit strength of 0.30, a recency effect of 0.50, and a reorder cost of negative 0.20.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   transitions = env.transition_matrices

   nfxp_result = NFXPEstimator(se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   ccp_result = CCPEstimator(num_policy_iterations=20, se_method="robust").estimate(panel, utility, env.problem_spec, transitions)
   mce_result = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.1, outer_max_iter=300)).estimate(panel, utility, env.problem_spec, transitions)

.. list-table:: Parameter Recovery (1600 consumers, 52 weeks)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP (K=20)
     - MCE-IRL
   * - habit_strength
     - 0.3000
     - 0.3869
     - 0.2024
     - 0.0651
   * - recency_effect
     - 0.5000
     - 0.5276
     - 0.2525
     - 0.5590
   * - reorder_cost
     - -0.2000
     - -0.3126
     - 0.2208
     - -0.0229

NFXP recovers all three parameters with the correct signs and reasonable magnitudes. The habit strength estimate of 0.387 is 29 percent above the true value of 0.30, the recency effect is within 6 percent, and the reorder cost is overestimated in magnitude at negative 0.31 versus negative 0.20. NFXP standard errors are 0.17 for habit strength, 0.06 for recency effect, and 0.15 for reorder cost. MCE-IRL recovers the recency effect accurately (0.559 versus 0.50) but underestimates habit strength and reorder cost. CCP misses the sign on reorder cost entirely, consistent with the NPL Hessian instabilities observed on other environments with asymmetric action frequencies.

Post-estimation diagnostics
---------------------------

The ``etable`` function places all three models side by side with significance stars.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

.. code-block:: text

   ==============================================================
                       NFXP (Nested Fixed Point)    NPL (K=20)MCE IRL (Ziebart 2010)
   ==============================================================
         habit_strength      0.3869**     0.2024***        0.0651
                             (0.1691)      (0.0000)      (0.1287)
         recency_effect     0.5276***     0.2525***     0.5590***
                             (0.0633)      (0.0000)      (0.0449)
           reorder_cost     -0.3126**     0.2208***       -0.0229
                             (0.1538)      (0.0000)      (0.1149)
   --------------------------------------------------------------
           Observations        83,200        83,200        83,200
         Log-Likelihood    -52,311.24    -52,324.20    -52,312.12
   ==============================================================

The Vuong test between NFXP and MCE-IRL yields a z-statistic of 0.636 with a p-value of 0.525, indicating the two models are statistically indistinguishable in fit. Brier scores are 0.437 for all three methods, and KL divergences are below 0.007, confirming that all three produce similar predictive accuracy despite the parameter differences.

.. list-table:: Fit Metrics
   :header-rows: 1

   * - Metric
     - NFXP
     - CCP
     - MCE-IRL
   * - Brier Score
     - 0.4370
     - 0.4372
     - 0.4370
   * - KL Divergence
     - 0.0048
     - 0.0062
     - 0.0056
   * - Log-Likelihood
     - -52,311
     - -52,324
     - -52,312

Counterfactual analysis
-----------------------

The free delivery promotion sets the reorder cost to zero, simulating a promotional offer that eliminates the hassle of placing an order. The welfare change is positive 4.45.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   new_params = nfxp_result.parameters.at[2].set(0.0)
   cf = counterfactual_policy(nfxp_result, new_params, utility, problem, transitions)

.. list-table:: Reorder Cost Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -100% (free)
     - +4.445
     - 0.070
   * - -50%
     - +2.170
     - 0.036
   * - -25%
     - +1.071
     - 0.018
   * - +25%
     - -1.043
     - 0.019
   * - +50%
     - -2.056
     - 0.039
   * - +100%
     - -3.991
     - 0.079

The elasticity is approximately symmetric around the baseline. Each 25 percent reduction in reorder cost raises welfare by about 1.1 and shifts the reorder probability by about 1.8 percentage points. The response is nearly linear across the range tested.

Running the example
-------------------

.. code-block:: bash

   python examples/instacart/run_estimation.py

Marketing interpretation
------------------------

This model connects to the Erdem and Keane (1996) tradition of structural brand choice models. The habit strength parameter measures state dependence, which marketing researchers use to separate true brand loyalty from spurious correlation driven by unobserved heterogeneity. The recency effect captures stockpiling and inventory management behavior. The reorder cost reflects switching costs and search frictions on the platform.

Counterfactual experiments are straightforward. Reducing the reorder cost (for example through a promotional free delivery offer) shifts the reorder rate upward. The structural model predicts not just the immediate effect but the long-run equilibrium through the transition dynamics, since more reorders today increase future purchase frequency which further increases future reorder rates.
