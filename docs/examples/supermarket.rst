Supermarket Pricing and Inventory
==================================

This example applies structural estimation to real supermarket data from Aguirregabiria (1999). A retailer manages 534 products in a single Spanish supermarket over 29 months. Each period the retailer makes two joint decisions for each product: whether to run a price promotion and whether to place an order from the supplier. Promotions boost sales volume but reduce margins. Orders replenish inventory but have logistical costs.

The data is discretized into 10 states (5 inventory quintile bins by 2 lagged promotion status levels) and 4 actions (promotion or regular price crossed with order or no order). Transitions are estimated directly from the 13,884 observed product-month transitions. Five features capture the economic structure: a holding cost proportional to inventory, a stockout indicator for critically low inventory without ordering, a net promotion effect, a lagged promotion indicator capturing demand persistence, and an order cost indicator.

.. image:: /_static/supermarket_overview.png
   :alt: Supermarket retail pricing and inventory management showing promotion frequency, inventory turnover, and order placement patterns across product categories.
   :width: 100%

Quick start
-----------

.. code-block:: python

   from econirl.environments.supermarket import SupermarketEnvironment
   from econirl.datasets.supermarket import load_supermarket

   env = SupermarketEnvironment(discount_factor=0.95)
   panel = load_supermarket(as_panel=True)

Estimation
----------

Three linear estimators and one neural reward estimator are fit on 427 training products (80 percent split) over 26 usable periods (11,102 observations).

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

.. list-table:: Estimation Results (427 products, 11,102 observations)
   :header-rows: 1

   * - Parameter
     - NFXP
     - NFXP SE
     - CCP (K=20)
     - MCE-IRL
     - MCE-IRL SE
   * - holding_cost
     - -1.5789
     - 0.0836
     - -1.5926
     - -1.5791
     - 0.1576
   * - stockout_penalty
     - -1.2917
     - 0.0734
     - -1.3150
     - -1.2917
     - 0.1408
   * - net_promotion_effect
     - 0.5271
     - 0.0000
     - -0.2892
     - 0.0813
     - 0.0363
   * - lagged_promotion
     - -0.3920
     - 0.0000
     - 0.4702
     - 0.0772
     - 0.0345
   * - order_cost
     - 1.7075
     - 0.0384
     - 1.7088
     - 1.7076
     - 0.1093

Three parameters are well identified and consistent across all estimators. The holding cost is negative 1.58, the stockout penalty is negative 1.29, and the order cost is 1.71. Adding the order cost feature improved the log-likelihood from negative 14,033 (3 features) to negative 13,218 (5 features) and prediction accuracy from 32.7 percent to 42.3 percent.

The order cost of 1.71 is positive, meaning the model assigns positive utility to ordering. This captures the fact that ordering is the predominant action in the data: retailers order frequently to maintain stock, and the "cost" of ordering is offset by the much larger benefit of avoiding stockouts. The structural interpretation is that the order cost parameter absorbs the combined effect of order logistics costs and inventory replenishment benefits, netting positive because replenishment benefits dominate.

The net_promotion_effect and lagged_promotion parameters disagree across estimators. NFXP estimates net_promotion_effect as 0.53 and lagged_promotion as negative 0.39, but CCP reverses both signs. MCE-IRL estimates both near zero (0.08 and 0.08) with marginal significance. This sign instability indicates that promotion dynamics and lagged promotion status are weakly identified in this data, likely because the 10-state discretization compresses too much variation in promotion frequency and timing.

Neural reward estimation
------------------------

A neural reward network (2 hidden layers of 64 units) learns a nonlinear R(s,a) via deep MCE-IRL.

.. code-block:: python

   from econirl.estimators.mceirl_neural import MCEIRLNeural

   neural = MCEIRLNeural(
       n_states=10, n_actions=4, discount=0.95,
       reward_type="state_action",
       reward_hidden_dim=64, reward_num_layers=2, max_epochs=200,
   )
   neural.fit(data=df, state="state", action="action", id="agent_id",
              features=env.feature_matrix, transitions=transitions)

The neural reward converges in 200 epochs with a projection R-squared of 0.66. The 5 linear features capture about two thirds of the neural reward surface, indicating that the linear specification is a reasonable but incomplete approximation. The remaining third of reward variation involves nonlinear patterns that the linear model cannot express, such as state-dependent ordering thresholds or promotion timing strategies that depend on specific inventory-promotion combinations.

Post-estimation diagnostics
---------------------------

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result, mce_result))

All three estimators achieve identical log-likelihoods at negative 13,218 and Brier scores at 0.644. Prediction accuracy is 42.3 percent against a random baseline of 25 percent for four actions.

Counterfactual analysis
-----------------------

The stockout tolerance experiment halves the stockout penalty, simulating a scenario where customers become more patient with out-of-stock items.

.. code-block:: python

   from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis
   stockout_idx = env.parameter_names.index("stockout_penalty")
   new_params = mce_result.parameters.at[stockout_idx].set(mce_result.parameters[stockout_idx] / 2)
   cf = counterfactual_policy(mce_result, new_params, utility, problem, transitions)

Halving the stockout penalty increases expected lifetime welfare by 0.35 utils.

.. list-table:: Stockout Penalty Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - +0.348
     - 0.011
   * - -25%
     - +0.143
     - 0.005
   * - +25%
     - -0.101
     - 0.003
   * - +50%
     - -0.174
     - 0.006
   * - +100%
     - -0.263
     - 0.009

If customers became more tolerant of empty shelves through substitute availability or online ordering, the retailer could afford to hold less inventory and run promotions more aggressively, capturing the 0.35 utils gain. The small policy changes (0.3 to 1.1 percent) indicate the retailer already manages inventory conservatively enough to avoid frequent stockouts, operating near the interior of the inaction region.

Running the example
-------------------

.. code-block:: bash

   python examples/supermarket/run_estimation.py

Retail IO interpretation
------------------------

The Aguirregabiria (1999) model demonstrates how dynamic optimization generates pricing patterns that static models cannot explain. The holding cost of negative 1.58 and stockout penalty of negative 1.29 are the dominant structural parameters, both well identified across all three estimators. The order cost of 1.71 is the new finding from the 5-feature specification: it captures the net benefit of replenishment and is the largest parameter in magnitude, reflecting the retailer's strong incentive to keep shelves stocked.

The neural reward R-squared of 0.66 suggests that about a third of the retailer's decision-making involves nonlinear patterns not captured by the 5 linear features. These could include inventory-level-dependent promotion thresholds (promote only when stock is high), seasonal ordering cycles, or supplier-driven promotion timing. A richer feature set or the neural reward itself could be used for more accurate counterfactual predictions when the 5-feature linear model is insufficient.

Reference
---------

Aguirregabiria, V. (1999). The Dynamics of Markups and Inventories in Retailing Firms. Review of Economic Studies, 66(2), 275-308.
